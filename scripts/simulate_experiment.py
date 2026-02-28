"""
Demo script: generate assignments, simulate outcomes, compute metrics, run reweight.
No external services; uses config seed for reproducibility.
Run from project root: python -m scripts.simulate_experiment
"""

import random
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config
from src.experiments.assign import assign_to_experiment
from src.experiments.storage import (
    read_assignments,
    read_outcomes,
    append_outcomes,
    read_pathway_metrics,
    write_pathway_metrics,
)
from src.experiments.outcome_simulator import simulate_outcome
from src.experiments.metrics import compute_pathway_metrics
from src.experiments.reweight import apply_uplift_reweighting


def make_demo_hypotheses():
    """Create a few minimal hypotheses for demo (eligible for experiment)."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "user_id": "demo_user_1",
            "persona_tier": "aspiring_affluent",
            "signal": "leapfrog_ready",
            "confidence": 0.82,
            "governance": {"tier": "green"},
            "traceability": {"target_product": {"code": "RRSP_LOAN", "name": "Retirement Accelerator"}},
            "staged_at": now,
        },
        {
            "user_id": "demo_user_2",
            "persona_tier": "aspiring_affluent",
            "signal": "leapfrog_ready",
            "confidence": 0.78,
            "governance": {"tier": "amber"},
            "traceability": {"target_product": {"code": "RRSP_LOAN", "name": "Retirement Accelerator"}},
            "staged_at": now,
        },
        {
            "user_id": "demo_user_3",
            "persona_tier": "sticky_family_leader",
            "signal": "liquidity_warning",
            "confidence": 0.75,
            "governance": {"tier": "green"},
            "traceability": {"target_product": {"code": "SUMMIT_PORTFOLIO", "name": "Summit Portfolio"}},
            "staged_at": now,
        },
        {
            "user_id": "demo_user_4",
            "persona_tier": "generation_nerd",
            "signal": "harvest_opportunity",
            "confidence": 0.88,
            "governance": {"tier": "amber"},
            "traceability": {"target_product": {"code": "AI_RESEARCH_DIRECT_INDEX", "name": "AI Research + Direct Index"}},
            "staged_at": now,
        },
    ]


def main():
    config = load_config()
    seed = config.get("experiment", {}).get("seed", 42)
    rng = random.Random(seed)

    print("1. Creating demo hypotheses and assigning to treatment/control...")
    hypotheses = make_demo_hypotheses()
    assignments = []
    for h in hypotheses:
        rec = assign_to_experiment(h, config, rng)
        if rec:
            assignments.append(rec)
            print(f"   Assigned {h['user_id']} -> {rec['assignment']}")
    print(f"   Total assignments: {len(assignments)}")

    print("2. Simulating outcomes for each assignment...")
    outcomes = []
    for rec in assignments:
        out = simulate_outcome(rec, {}, {}, config, rng)
        outcomes.append(out)
    append_outcomes(outcomes)
    print(f"   Outcomes written: {len(outcomes)}")

    print("3. Computing pathway metrics...")
    assignments_df = read_assignments()
    outcomes_df = read_outcomes()
    metrics_df = compute_pathway_metrics(assignments_df, outcomes_df, config)
    write_pathway_metrics(metrics_df)
    print(f"   Pathways: {len(metrics_df)}")
    if not metrics_df.empty:
        print(metrics_df[["persona_tier", "signal", "product_code", "n_treatment", "n_control", "uplift_score"]].to_string())

    print("4. Reweighting a sample hypothesis...")
    sample = hypotheses[0]
    priority_score, safety_actions = apply_uplift_reweighting(sample, metrics_df, config)
    print(f"   Priority score: {priority_score:.4f}")
    print(f"   Safety actions: {safety_actions}")

    print("Done. Run the UI or GET /experiment/metrics to see results.")


if __name__ == "__main__":
    main()
