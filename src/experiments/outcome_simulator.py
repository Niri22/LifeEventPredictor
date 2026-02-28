"""Simulate downstream outcomes for experiment assignments (deterministic with config + RNG)."""

from datetime import datetime, timezone
import uuid


def _pathway_key(persona_tier: str, signal: str, product_code: str) -> str:
    return f"{persona_tier}|{signal}|{product_code}"


def simulate_outcome(
    assignment_record: dict,
    user_features: dict,
    macro_context: dict | None,
    config: dict,
    rng,
) -> dict:
    """
    Simulate outcome for one assignment. Control gets baseline + noise; treatment gets baseline + lift + noise.
    Macro can modulate (e.g. high VIX reduces PE outcome lifts).
    """
    exp = config.get("experiment", {})
    effects_map = exp.get("effects", {})
    observation_window_days = exp.get("observation_window_days", 90)

    persona_tier = assignment_record.get("persona_tier", "")
    signal = assignment_record.get("signal", "")
    product_code = assignment_record.get("product_code", "")
    is_treatment = (assignment_record.get("assignment") or "").lower() == "treatment"

    key = _pathway_key(persona_tier, signal, product_code)
    effects = effects_map.get(key, {})

    # Baseline (small random outcomes for both arms)
    base_convert = 0.15 + rng.random() * 0.15
    base_aua = (rng.random() - 0.3) * 2000
    base_liq = (rng.random() - 0.2) * 0.5
    base_retain = 0.92 + rng.random() * 0.06
    base_complaint = rng.random() * 0.02

    if is_treatment and effects:
        # Apply lift + noise
        conv_lift = effects.get("conversion_lift", 0) * (0.8 + rng.random() * 0.4)
        aua_lift = effects.get("aua_lift", 0) * (0.7 + rng.random() * 0.6)
        liq_lift = effects.get("liquidity_months_lift", 0) * (0.8 + rng.random() * 0.4)
        ret_lift = effects.get("retention_lift", 0) * (0.5 + rng.random())
        comp_lift = effects.get("complaint_lift", 0) * (0.5 + rng.random())

        # Macro modulation: high VIX reduces PE-related outcomes
        vix = (macro_context or {}).get("vix", 18)
        if product_code == "SUMMIT_PORTFOLIO" and vix > 25:
            aua_lift *= 0.5
            conv_lift *= 0.7
        base_convert += conv_lift
        base_aua += aua_lift
        base_liq += liq_lift
        base_retain += ret_lift
        base_complaint += comp_lift

    converted = 1 if rng.random() < base_convert else 0
    retained = 1 if rng.random() < base_retain else 0
    complaint = 1 if rng.random() < base_complaint else 0

    return {
        "outcome_id": str(uuid.uuid4()),
        "assignment_id": assignment_record["assignment_id"],
        "hypothesis_id": assignment_record.get("hypothesis_id", ""),
        "user_id": assignment_record["user_id"],
        "observation_window_days": observation_window_days,
        "converted": converted,
        "delta_aua": base_aua,
        "delta_liquidity_months": base_liq,
        "retained": retained,
        "complaint": complaint,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
