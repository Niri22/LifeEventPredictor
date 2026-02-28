"""Apply uplift reweighting to hypothesis priority score; apply safety brakes when outcomes are harmful."""


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def apply_uplift_reweighting(hypothesis: dict, metrics_df, config: dict) -> tuple[float, list[dict]]:
    """
    Compute priority_score = calibrated_confidence * (1 + uplift_weight), with uplift_weight clamped.
    If pathway has significance_flag, use uplift_score to derive uplift_weight; else 0.
    Safety brake: if retention_uplift, complaint_uplift, or liquidity_uplift cross thresholds,
    append safety_actions (e.g. downgrade governance, requires_human_review).
    Returns (priority_score, safety_actions).
    """
    exp = config.get("experiment", {})
    clamp_lo = float(exp.get("clamp_bounds", {}).get("uplift_weight_min", -0.25))
    clamp_hi = float(exp.get("clamp_bounds", {}).get("uplift_weight_max", 0.20))
    safety = exp.get("safety_thresholds", {})
    retention_min = float(safety.get("retention_uplift_min", -0.05))
    complaint_max = float(safety.get("complaint_uplift_max", 0.03))
    liquidity_min = float(safety.get("liquidity_uplift_min", -0.10))

    calibrated = float(hypothesis.get("confidence", 0))
    product_code = (hypothesis.get("traceability") or {}).get("target_product", {}).get("code", "")
    persona_tier = hypothesis.get("persona_tier", "")
    signal = hypothesis.get("signal", "")

    uplift_weight = 0.0
    safety_actions = []

    if metrics_df is not None and not metrics_df.empty:
        mask = (
            (metrics_df["persona_tier"] == persona_tier)
            & (metrics_df["signal"] == signal)
            & (metrics_df["product_code"] == product_code)
        )
        match = metrics_df.loc[mask]
        if not match.empty and match.iloc[0].get("significance_flag"):
            raw_score = match.iloc[0].get("uplift_score", 0) or 0
            # Map uplift_score (can be any magnitude) to weight: e.g. tanh-like scaling
            uplift_weight = _clamp(raw_score * 2.0, clamp_lo, clamp_hi)

        if not match.empty:
            row = match.iloc[0]
            ret_uplift = row.get("retention_uplift", 0) or 0
            comp_uplift = row.get("complaint_uplift", 0) or 0
            liq_uplift = row.get("liquidity_uplift", 0) or 0
            if ret_uplift < retention_min:
                safety_actions.append({
                    "type": "retention_harm", "metric": "retention_uplift", "value": ret_uplift,
                    "action": "downgrade_governance", "reason": f"Retention uplift {ret_uplift:.3f} below threshold {retention_min}",
                })
            if comp_uplift > complaint_max:
                safety_actions.append({
                    "type": "complaint_rise", "metric": "complaint_uplift", "value": comp_uplift,
                    "action": "requires_human_review", "reason": f"Complaint uplift {comp_uplift:.3f} above threshold {complaint_max}",
                })
            if liq_uplift < liquidity_min:
                safety_actions.append({
                    "type": "liquidity_stress", "metric": "liquidity_uplift", "value": liq_uplift,
                    "action": "downgrade_governance", "reason": f"Liquidity uplift {liq_uplift:.3f} below threshold {liquidity_min}",
                })

    priority_score = calibrated * (1.0 + uplift_weight)
    return (priority_score, safety_actions)
