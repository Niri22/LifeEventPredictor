"""Experiment endpoints: assign, simulate, metrics, reweight."""

import random
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter

from api.schemas import (
    ExperimentAssignRequest,
    ExperimentAssignResponse,
    ExperimentAssignmentRecord,
    ExperimentSimulateRequest,
    ExperimentSimulateResponse,
    ExperimentMetricsResponse,
    PathwayMetricRow,
    ExperimentReweightRequest,
    ExperimentReweightResponse,
)
from src.utils.io import load_config
from src.experiments.assign import assign_to_experiment
from src.experiments.storage import read_assignments, read_outcomes, append_outcomes, write_pathway_metrics
from src.experiments.outcome_simulator import simulate_outcome
from src.experiments.metrics import compute_pathway_metrics
from src.experiments.reweight import apply_uplift_reweighting

router = APIRouter(prefix="/experiment", tags=["experiment"])


def _get_rng(config: dict):
    seed = config.get("experiment", {}).get("seed", 42)
    return random.Random(seed)


@router.post("/assign", response_model=ExperimentAssignResponse)
def experiment_assign(req: ExperimentAssignRequest):
    """Assign hypothesis to treatment or control if eligible. Appends to experiment_assignments.parquet."""
    config = load_config()
    rng = _get_rng(config)
    hypothesis = {
        "user_id": req.user_id,
        "persona_tier": req.persona_tier,
        "signal": req.signal,
        "confidence": req.confidence,
        "governance": req.governance,
        "traceability": req.traceability,
        "hypothesis_id": req.hypothesis_id,
        "staged_at": req.staged_at or datetime.now(timezone.utc).isoformat(),
    }
    record = assign_to_experiment(hypothesis, config, rng)
    if record is None:
        return ExperimentAssignResponse(assignment=None)
    return ExperimentAssignResponse(
        assignment=ExperimentAssignmentRecord(
            assignment_id=record["assignment_id"],
            hypothesis_id=record["hypothesis_id"],
            user_id=record["user_id"],
            persona_tier=record["persona_tier"],
            signal=record["signal"],
            product_code=record["product_code"],
            governance_tier_at_assignment=record["governance_tier_at_assignment"],
            calibrated_confidence_at_assignment=record["calibrated_confidence_at_assignment"],
            assignment=record["assignment"],
            created_at=record["created_at"],
            experiment_version=record["experiment_version"],
        )
    )


@router.post("/simulate", response_model=ExperimentSimulateResponse)
def experiment_simulate(req: ExperimentSimulateRequest):
    """Simulate outcomes for assignments. If assignment_ids given, simulate those; else assignments older than observation_window_days."""
    config = load_config()
    rng = _get_rng(config)
    exp = config.get("experiment", {})
    window_days = int(exp.get("observation_window_days", 90))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

    assignments_df = read_assignments()
    if assignments_df.empty:
        return ExperimentSimulateResponse(outcomes_written=0)

    if req.assignment_ids:
        assignments_df = assignments_df[assignments_df["assignment_id"].isin(req.assignment_ids)]
    else:
        assignments_df = assignments_df[assignments_df["created_at"] < cutoff]

    # Skip assignments that already have outcomes (idempotent)
    existing_outcomes = read_outcomes()
    if not existing_outcomes.empty and "assignment_id" in existing_outcomes.columns:
        done_ids = set(existing_outcomes["assignment_id"].tolist())
        assignments_df = assignments_df[~assignments_df["assignment_id"].isin(done_ids)]

    outcomes = []
    for _, row in assignments_df.iterrows():
        assignment_record = row.to_dict()
        outcome = simulate_outcome(
            assignment_record,
            user_features={},
            macro_context={},
            config=config,
            rng=rng,
        )
        outcomes.append(outcome)

    if outcomes:
        append_outcomes(outcomes)
    return ExperimentSimulateResponse(outcomes_written=len(outcomes))


@router.get("/metrics", response_model=ExperimentMetricsResponse)
def experiment_metrics():
    """Return pathway metrics (recomputed from assignments + outcomes)."""
    config = load_config()
    assignments_df = read_assignments()
    outcomes_df = read_outcomes()
    metrics_df = compute_pathway_metrics(assignments_df, outcomes_df, config)
    if metrics_df.empty:
        return ExperimentMetricsResponse(pathways=[])

    pathways = [
        PathwayMetricRow(
            persona_tier=row["persona_tier"],
            signal=row["signal"],
            product_code=row["product_code"],
            experiment_version=row["experiment_version"],
            n_treatment=int(row["n_treatment"]),
            n_control=int(row["n_control"]),
            conversion_uplift=float(row["conversion_uplift"]),
            delta_aua_uplift=float(row["delta_aua_uplift"]),
            liquidity_uplift=float(row["liquidity_uplift"]),
            retention_uplift=float(row["retention_uplift"]),
            complaint_uplift=float(row["complaint_uplift"]),
            uplift_score=float(row["uplift_score"]),
            significance_flag=bool(row["significance_flag"]),
            last_updated_at=row["last_updated_at"],
        )
        for _, row in metrics_df.iterrows()
    ]
    return ExperimentMetricsResponse(pathways=pathways)


@router.post("/reweight", response_model=ExperimentReweightResponse)
def experiment_reweight(req: ExperimentReweightRequest):
    """Return priority_score and safety_actions for a hypothesis using pathway uplift."""
    config = load_config()
    assignments_df = read_assignments()
    outcomes_df = read_outcomes()
    metrics_df = compute_pathway_metrics(assignments_df, outcomes_df, config)

    hypothesis = {
        "user_id": req.user_id,
        "persona_tier": req.persona_tier,
        "signal": req.signal,
        "confidence": req.confidence,
        "governance": req.governance,
        "traceability": req.traceability,
    }
    priority_score, safety_actions = apply_uplift_reweighting(hypothesis, metrics_df, config)

    explanation_parts = []
    if safety_actions:
        explanation_parts = [a.get("reason", a.get("type", "")) for a in safety_actions]
    explanation = "; ".join(explanation_parts) if explanation_parts else "Uplift reweighting applied from pathway metrics."

    return ExperimentReweightResponse(
        priority_score=priority_score,
        safety_actions=safety_actions,
        explanation=explanation,
    )
