"""Assign eligible hypotheses to treatment or control."""

import uuid
from datetime import datetime, timezone

from src.experiments.storage import append_assignments


def assign_to_experiment(hypothesis: dict, config: dict, rng) -> dict | None:
    """
    Assign hypothesis to treatment or control if eligible.
    Eligibility: governance_tier in allow_governance_tiers, product_code in allow_products.
    Returns assignment record dict or None if not eligible. Side-effect: appends to experiment_assignments.parquet.
    """
    exp = config.get("experiment", {})
    allow_tiers = set(exp.get("allow_governance_tiers", ["green", "amber"]))
    allow_products = set(exp.get("allow_products", []))
    treatment_ratio = float(exp.get("treatment_ratio", 0.8))
    experiment_version = exp.get("experiment_version", "v1")

    product_code = (hypothesis.get("traceability") or {}).get("target_product", {}).get("code", "")
    gov = hypothesis.get("governance") or {}
    tier = (gov.get("tier") or "").lower()

    if tier not in allow_tiers or product_code not in allow_products:
        return None

    hypothesis_id = hypothesis.get("hypothesis_id") or str(
        uuid.uuid5(uuid.NAMESPACE_DNS, f"{hypothesis['user_id']}|{hypothesis.get('signal', '')}|{product_code}|{hypothesis.get('staged_at', '')}")
    )
    assignment = "treatment" if rng.random() < treatment_ratio else "control"
    created_at = datetime.now(timezone.utc).isoformat()

    record = {
        "assignment_id": str(uuid.uuid4()),
        "hypothesis_id": hypothesis_id,
        "user_id": hypothesis["user_id"],
        "persona_tier": hypothesis.get("persona_tier", ""),
        "signal": hypothesis.get("signal", ""),
        "product_code": product_code,
        "governance_tier_at_assignment": tier,
        "calibrated_confidence_at_assignment": float(hypothesis.get("confidence", 0)),
        "assignment": assignment,
        "created_at": created_at,
        "experiment_version": experiment_version,
    }
    append_assignments([record])
    return record
