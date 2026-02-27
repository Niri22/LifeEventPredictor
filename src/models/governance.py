"""
Tiered Governance Engine -- Traffic Light classification.

Assigns each hypothesis to a governance tier based on model confidence
and product risk profile, enabling differentiated approval workflows:
  - GREEN:  Low risk / high confidence (>0.9)  -> silent approval / notification
  - AMBER:  Medium risk / mid confidence (0.7-0.9) -> batch cohort approval
  - RED:    High risk / high stakes -> 1-to-1 human review
"""

from __future__ import annotations

PRODUCT_RISK = {
    "RRSP_LOAN": "medium",
    "SUMMIT_PORTFOLIO": "high",
    "AI_RESEARCH_DIRECT_INDEX": "medium",
}

ILLIQUIDITY_THRESHOLD = 0.20  # Summit allocation > 20% of AUA triggers RED


def classify_governance_tier(
    confidence: float,
    product_code: str,
    illiquidity_ratio: float = 0.0,
    aua_current: float = 0.0,
) -> dict:
    """
    Return governance tier and reasoning.

    Returns:
        {
            "tier": "green" | "amber" | "red",
            "label": "Green - Auto-Approve" | ...,
            "reason": str,
            "workflow": str,
        }
    """
    risk = PRODUCT_RISK.get(product_code, "medium")

    # RED: high-stakes products with dangerous allocation or very low confidence
    if risk == "high" and illiquidity_ratio > ILLIQUIDITY_THRESHOLD:
        return {
            "tier": "red",
            "label": "Red - Manual Review Required",
            "reason": (
                f"Summit Portfolio allocation ({illiquidity_ratio:.0%} of AUA) "
                f"exceeds {ILLIQUIDITY_THRESHOLD:.0%} safety threshold."
            ),
            "workflow": "1-to-1 human review",
        }

    if confidence < 0.60:
        return {
            "tier": "red",
            "label": "Red - Low Confidence",
            "reason": f"Model confidence {confidence:.1%} is below the minimum surfacing threshold.",
            "workflow": "1-to-1 human review",
        }

    # GREEN: high confidence + acceptable risk
    if confidence > 0.90 and risk != "high":
        return {
            "tier": "green",
            "label": "Green - Auto-Approve Candidate",
            "reason": f"High confidence ({confidence:.1%}) with {risk}-risk product.",
            "workflow": "Silent approval or low-friction notification",
        }

    # AMBER: everything else
    return {
        "tier": "amber",
        "label": "Amber - Batch Review",
        "reason": f"Confidence {confidence:.1%} with {risk}-risk product; suitable for cohort batch approval.",
        "workflow": "Batch cohort approval",
    }


def enrich_hypothesis_with_governance(hypothesis: dict) -> dict:
    """Add governance tier info to an existing hypothesis dict in-place and return it."""
    conf = hypothesis.get("confidence", 0.0)
    product_code = hypothesis.get("traceability", {}).get("target_product", {}).get("code", "")

    # Pull illiquidity_ratio from audit log if available
    illiquidity = 0.0
    for entry in hypothesis.get("traceability", {}).get("audit_log", []):
        if entry.get("feature") == "illiquidity_ratio":
            illiquidity = entry.get("value", 0.0)
            break

    aua = 0.0
    for entry in hypothesis.get("traceability", {}).get("audit_log", []):
        if entry.get("feature") == "aua_current":
            aua = entry.get("value", 0.0)
            break

    gov = classify_governance_tier(conf, product_code, illiquidity, aua)
    hypothesis["governance"] = gov
    return hypothesis
