"""Tests for the experimentation layer: assign, metrics, reweight, API."""

import random
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.io import load_config
from src.experiments.assign import assign_to_experiment
from src.experiments.metrics import compute_pathway_metrics
from src.experiments.reweight import apply_uplift_reweighting


# ---------------------------------------------------------------------------
# Assign
# ---------------------------------------------------------------------------
def test_assign_eligible_returns_record(tmp_path):
    """Eligible hypothesis (green tier, allow_products) returns assignment record."""
    config = load_config()
    config["experiment"] = config.get("experiment", {})
    config["experiment"]["allow_governance_tiers"] = ["green", "amber"]
    config["experiment"]["allow_products"] = ["RRSP_LOAN"]
    config["experiment"]["treatment_ratio"] = 0.5
    config["experiment"]["experiment_version"] = "v1"

    hypothesis = {
        "user_id": "u1",
        "persona_tier": "aspiring_affluent",
        "signal": "leapfrog_ready",
        "confidence": 0.8,
        "governance": {"tier": "green"},
        "traceability": {"target_product": {"code": "RRSP_LOAN", "name": "RRSP Loan"}},
        "staged_at": "2025-01-01T00:00:00Z",
    }
    rng = random.Random(42)
    with patch("src.experiments.assign.append_assignments"):
        record = assign_to_experiment(hypothesis, config, rng)
    assert record is not None
    assert record["user_id"] == "u1"
    assert record["assignment"] in ("treatment", "control")
    assert record["product_code"] == "RRSP_LOAN"
    assert record["governance_tier_at_assignment"] == "green"


def test_assign_ineligible_returns_none():
    """Ineligible hypothesis (e.g. red tier or product not in list) returns None."""
    config = load_config()
    config["experiment"] = config.get("experiment", {})
    config["experiment"]["allow_governance_tiers"] = ["green", "amber"]
    config["experiment"]["allow_products"] = ["RRSP_LOAN"]

    hypothesis_red = {
        "user_id": "u2",
        "persona_tier": "aspiring_affluent",
        "signal": "leapfrog_ready",
        "confidence": 0.8,
        "governance": {"tier": "red"},
        "traceability": {"target_product": {"code": "RRSP_LOAN"}},
        "staged_at": "2025-01-01T00:00:00Z",
    }
    rng = random.Random(42)
    with patch("src.experiments.assign.append_assignments"):
        assert assign_to_experiment(hypothesis_red, config, rng) is None

    hypothesis_bad_product = {
        "user_id": "u3",
        "persona_tier": "aspiring_affluent",
        "signal": "leapfrog_ready",
        "confidence": 0.8,
        "governance": {"tier": "green"},
        "traceability": {"target_product": {"code": "OTHER_PRODUCT"}},
        "staged_at": "2025-01-01T00:00:00Z",
    }
    with patch("src.experiments.assign.append_assignments"):
        assert assign_to_experiment(hypothesis_bad_product, config, rng) is None


def test_assign_reproducible_with_fixed_seed():
    """With fixed seed, same hypothesis yields same assignment (deterministic)."""
    config = load_config()
    config["experiment"] = config.get("experiment", {})
    config["experiment"]["allow_governance_tiers"] = ["green"]
    config["experiment"]["allow_products"] = ["RRSP_LOAN"]
    config["experiment"]["treatment_ratio"] = 0.5
    config["experiment"]["experiment_version"] = "v1"

    hypothesis = {
        "user_id": "fixed_user",
        "persona_tier": "aspiring_affluent",
        "signal": "leapfrog_ready",
        "confidence": 0.8,
        "governance": {"tier": "green"},
        "traceability": {"target_product": {"code": "RRSP_LOAN"}},
        "staged_at": "2025-01-01T00:00:00Z",
    }
    with patch("src.experiments.assign.append_assignments"):
        rng1 = random.Random(12345)
        rec1 = assign_to_experiment(hypothesis, config, rng1)
        rng2 = random.Random(12345)
        rec2 = assign_to_experiment(hypothesis, config, rng2)
    assert rec1["assignment"] == rec2["assignment"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_compute_pathway_metrics_returns_expected_columns():
    """compute_pathway_metrics returns DataFrame with expected columns and uplift logic."""
    assignments_df = pd.DataFrame([
        {"assignment_id": "a1", "persona_tier": "p1", "signal": "s1", "product_code": "RRSP", "assignment": "treatment", "experiment_version": "v1"},
        {"assignment_id": "a2", "persona_tier": "p1", "signal": "s1", "product_code": "RRSP", "assignment": "control", "experiment_version": "v1"},
    ])
    outcomes_df = pd.DataFrame([
        {"assignment_id": "a1", "converted": 1, "delta_aua": 3000, "delta_liquidity_months": 0.5, "retained": 1, "complaint": 0},
        {"assignment_id": "a2", "converted": 0, "delta_aua": 500, "delta_liquidity_months": 0.1, "retained": 1, "complaint": 0},
    ])
    config = load_config()
    config["experiment"] = config.get("experiment", {})
    config["experiment"]["min_n"] = 2

    result = compute_pathway_metrics(assignments_df, outcomes_df, config)
    assert not result.empty
    assert "persona_tier" in result.columns
    assert "signal" in result.columns
    assert "product_code" in result.columns
    assert "n_treatment" in result.columns
    assert "n_control" in result.columns
    assert "conversion_uplift" in result.columns
    assert "delta_aua_uplift" in result.columns
    assert "uplift_score" in result.columns
    assert "significance_flag" in result.columns
    row = result.iloc[0]
    assert row["conversion_uplift"] == 1.0 - 0.0
    assert row["delta_aua_uplift"] == 3000 - 500
    assert row["significance_flag"] == True  # numpy bool


# ---------------------------------------------------------------------------
# Reweighting
# ---------------------------------------------------------------------------
def test_reweight_clamp_within_bounds():
    """uplift_weight is clamped to config bounds; priority_score reflects it."""
    config = load_config()
    config["experiment"] = config.get("experiment", {})
    config["experiment"]["clamp_bounds"] = {"uplift_weight_min": -0.25, "uplift_weight_max": 0.20}
    config["experiment"]["safety_thresholds"] = {"retention_uplift_min": -0.05, "complaint_uplift_max": 0.03, "liquidity_uplift_min": -0.10}

    hypothesis = {"user_id": "u", "persona_tier": "p1", "signal": "s1", "confidence": 0.8, "traceability": {"target_product": {"code": "RRSP"}}}
    # Very high uplift_score would yield > 0.2 weight if unclamped
    metrics_df = pd.DataFrame([{
        "persona_tier": "p1", "signal": "s1", "product_code": "RRSP",
        "uplift_score": 2.0, "significance_flag": True,
        "retention_uplift": 0.01, "complaint_uplift": 0.0, "liquidity_uplift": 0.05,
    }])
    priority_score, safety_actions = apply_uplift_reweighting(hypothesis, metrics_df, config)
    # priority = 0.8 * (1 + 0.2) = 0.96 (clamped to 0.2)
    assert priority_score <= 1.0
    assert 0.9 <= priority_score <= 1.0


def test_reweight_safety_brake_triggers():
    """When retention_uplift below threshold, safety_actions non-empty with reason."""
    config = load_config()
    config["experiment"] = config.get("experiment", {})
    config["experiment"]["clamp_bounds"] = {"uplift_weight_min": -0.25, "uplift_weight_max": 0.20}
    config["experiment"]["safety_thresholds"] = {"retention_uplift_min": -0.05, "complaint_uplift_max": 0.03, "liquidity_uplift_min": -0.10}

    hypothesis = {"user_id": "u", "persona_tier": "p1", "signal": "s1", "confidence": 0.8, "traceability": {"target_product": {"code": "RRSP"}}}
    metrics_df = pd.DataFrame([{
        "persona_tier": "p1", "signal": "s1", "product_code": "RRSP",
        "uplift_score": 0.02, "significance_flag": True,
        "retention_uplift": -0.10,  # below -0.05
        "complaint_uplift": 0.0, "liquidity_uplift": 0.0,
    }])
    priority_score, safety_actions = apply_uplift_reweighting(hypothesis, metrics_df, config)
    assert len(safety_actions) >= 1
    assert any(a.get("type") == "retention_harm" or "retention" in str(a.get("reason", "")).lower() for a in safety_actions)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_experiment_assign_endpoint_schema():
    """POST /experiment/assign returns 200 and expected schema (assignment or null)."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    payload = {
        "user_id": "api_user",
        "persona_tier": "aspiring_affluent",
        "signal": "leapfrog_ready",
        "confidence": 0.85,
        "governance": {"tier": "green"},
        "traceability": {"target_product": {"code": "RRSP_LOAN", "name": "RRSP Loan"}},
    }
    r = client.post("/experiment/assign", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "assignment" in data
    if data["assignment"] is not None:
        assert "assignment_id" in data["assignment"]
        assert data["assignment"]["assignment"] in ("treatment", "control")


def test_experiment_metrics_endpoint_schema():
    """GET /experiment/metrics returns 200 and list of pathway metrics."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.get("/experiment/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "pathways" in data
    assert isinstance(data["pathways"], list)
