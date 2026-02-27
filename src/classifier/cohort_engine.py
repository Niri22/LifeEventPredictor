"""
Cohort Engine by Strategic Intent.

Groups users/hypotheses into intent-based cohorts for Batch Review cards:
- Premium Leapfrog: High income, <$100k AUA, high RRSP room (Momentum Builder)
- Summit Onboarding: Premium status, low PE allocation
- Bank-Replacement Lead: High CC spend, no Direct Deposit at WS
"""

from __future__ import annotations

PREMIUM_AUA_MAX = 100_000
LEGACY_AUA_MIN = 100_000
SUMMIT_CC_PCT = 0.90


def build_intent_cohorts(hypotheses: list[dict]) -> list[dict]:
    """
    Group hypotheses by strategic intent.

    Returns list of:
      {
        "intent_id": str,
        "name": str,
        "why_summary": str,
        "potential_aum_growth": float | None,
        "hypotheses": list[dict],
        "user_ids": list[str],
      }
    """
    premium_leapfrog = []
    summit_onboarding = []
    bank_replacement = []

    for h in hypotheses:
        persona = h.get("persona_tier", "")
        aua = 0.0
        for e in h.get("traceability", {}).get("audit_log", []):
            if e.get("feature") == "aua_current":
                aua = e.get("value", 0.0)
                break
        illiquidity = 0.0
        for e in h.get("traceability", {}).get("audit_log", []):
            if e.get("feature") == "illiquidity_ratio":
                illiquidity = e.get("value", 0.0)
                break
        product_code = h.get("traceability", {}).get("target_product", {}).get("code", "")
        dist = h.get("distance_to_upgrade") or {}
        gap = dist.get("gap_dollars", 0)

        # Premium Leapfrog: Momentum Builder, AUA < 100k, RRSP product
        if persona == "aspiring_affluent" and aua < PREMIUM_AUA_MAX and product_code == "RRSP_LOAN":
            premium_leapfrog.append(h)

        # Summit Onboarding: Premium (AUA >= 100k), low illiquidity, Summit product
        if aua >= LEGACY_AUA_MIN and illiquidity < 0.20 and product_code == "SUMMIT_PORTFOLIO":
            summit_onboarding.append(h)

        # Bank-Replacement Lead: flag on hypothesis from guardrails
        if h.get("bank_replacement_lead"):
            bank_replacement.append(h)

    cohorts = []

    if premium_leapfrog:
        aum_growth = sum((h.get("distance_to_upgrade") or {}).get("gap_dollars", 0) for h in premium_leapfrog)
        cohorts.append({
            "intent_id": "premium_leapfrog",
            "name": "Premium Leapfrog",
            "why_summary": f"${aum_growth:,.0f} potential AUM growth via RRSP loans to reach $100k Premium status.",
            "potential_aum_growth": aum_growth,
            "hypotheses": premium_leapfrog,
            "user_ids": [h["user_id"] for h in premium_leapfrog],
        })

    if summit_onboarding:
        cohorts.append({
            "intent_id": "summit_onboarding",
            "name": "Summit Onboarding",
            "why_summary": "Premium clients with low PE allocation; institutional access with liquidity safeguard.",
            "potential_aum_growth": None,
            "hypotheses": summit_onboarding,
            "user_ids": [h["user_id"] for h in summit_onboarding],
        })

    if bank_replacement:
        cohorts.append({
            "intent_id": "bank_replacement_lead",
            "name": "Bank-Replacement Lead",
            "why_summary": "High WS Credit Card spend, Direct Deposit elsewhere; nudge to move DD for Cash interest.",
            "potential_aum_growth": None,
            "hypotheses": bank_replacement,
            "user_ids": [h["user_id"] for h in bank_replacement],
        })

    return cohorts
