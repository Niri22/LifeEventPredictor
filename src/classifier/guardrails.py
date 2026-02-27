"""
Intelligence Guardrails: anomaly detection and risk cohorts.

A. Outlier Sentinel: Life Inflection Alert when HNW users show a large spike
   in everyday (grocery/retail) spend — avoid suggesting high-margin loans.
B. Cross-Pollination: Bank-Replacement Lead when high WS CC spend but no
   Direct Deposit at WS.
C. Liquidity Stress: Flag Summit users with runway < 3 months for curator
   to pause automated PE contributions.
"""

from __future__ import annotations

OUTLIER_SPIKE_RATIO = 4.0  # 400% MoM = Life Inflection Alert
LIQUIDITY_RUNWAY_MONTHS = 3.0
BANK_REPLACEMENT_CC_PCT = 0.90


def check_outlier_sentinel(
    user_features: dict,
    persona_tier: str,
    spike_ratio_threshold: float = OUTLIER_SPIKE_RATIO,
) -> dict:
    """
    For Legacy Architect (HNW), detect a large MoM spike in everyday spend.
    Returns {"alert": bool, "reason": str}. If alert is True, caller should
    set life_inflection_alert on hypothesis and exclude/demote RRSP Loan.
    """
    if persona_tier != "generation_nerd":
        return {"alert": False, "reason": ""}
    ratio = user_features.get("everyday_spend_mom_ratio", 1.0)
    if ratio >= spike_ratio_threshold:
        return {
            "alert": True,
            "reason": (
                f"Life Inflection Alert: {ratio:.0%} MoM spike in grocery/retail spend. "
                "Cash flow may be unstable; do not suggest high-margin loans."
            ),
        }
    return {"alert": False, "reason": ""}


def check_cross_pollination(
    user_features: dict,
    pct_threshold: float = BANK_REPLACEMENT_CC_PCT,
) -> dict:
    """
    User uses WS Credit Card for most spend but has Direct Deposit elsewhere.
    Returns {"in_cohort": bool, "reason": str}. If in_cohort, they are a
    "Primary Bank Conversion" / Bank-Replacement Lead.
    """
    pct_cc = user_features.get("pct_spend_on_ws_cc", 0.0)
    dd_at_ws = user_features.get("direct_deposit_at_ws", False)
    if pct_cc >= pct_threshold and not dd_at_ws:
        return {
            "in_cohort": True,
            "reason": (
                f"Bank-Replacement Lead: {pct_cc:.0%} of spend on WS Credit Card "
                "but Direct Deposit at another institution. Nudge: move DD to unlock Cash interest."
            ),
        }
    return {"in_cohort": False, "reason": ""}


def check_liquidity_stress(
    months_of_runway: float,
    product_code: str,
    runway_threshold: float = LIQUIDITY_RUNWAY_MONTHS,
) -> dict:
    """
    For Summit (PE) users, flag when liquidity buffer is below threshold.
    Returns {"in_cohort": bool, "reason": str}. Curator can pause PE contributions.
    """
    if product_code != "SUMMIT_PORTFOLIO":
        return {"in_cohort": False, "reason": ""}
    if months_of_runway < runway_threshold:
        return {
            "in_cohort": True,
            "reason": (
                f"Liquidity Risk: {months_of_runway:.1f} months runway below "
                f"{runway_threshold} months. Consider pausing automated PE contributions."
            ),
        }
    return {"in_cohort": False, "reason": ""}


def apply_guardrails_to_hypothesis(hypothesis: dict, features: dict) -> dict:
    """
    Run all guardrails and attach flags/reasons to the hypothesis.
    - life_inflection_alert: bool (Outlier Sentinel)
    - bank_replacement_lead: bool (Cross-Pollination)
    - liquidity_stress: bool (Liquidity Stress)
    - guardrail_reasons: list[str]
    If life_inflection_alert is True, caller may demote or hide RRSP Loan.
    """
    hypothesis["life_inflection_alert"] = False
    hypothesis["bank_replacement_lead"] = False
    hypothesis["liquidity_stress"] = False
    hypothesis["guardrail_reasons"] = []

    out = check_outlier_sentinel(features, hypothesis.get("persona_tier", ""))
    if out["alert"]:
        hypothesis["life_inflection_alert"] = True
        hypothesis["guardrail_reasons"].append(out["reason"])

    cp = check_cross_pollination(features)
    if cp["in_cohort"]:
        hypothesis["bank_replacement_lead"] = True
        hypothesis["guardrail_reasons"].append(cp["reason"])

    runway = hypothesis.get("traceability", {}).get("spending_buffer", {}).get("months_of_runway", 12.0)
    prod_code = hypothesis.get("traceability", {}).get("target_product", {}).get("code", "")
    liq = check_liquidity_stress(runway, prod_code)
    if liq["in_cohort"]:
        hypothesis["liquidity_stress"] = True
        hypothesis["guardrail_reasons"].append(liq["reason"])

    return hypothesis
