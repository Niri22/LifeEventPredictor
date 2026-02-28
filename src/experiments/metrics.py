"""Compute pathway-level uplift metrics from assignments and outcomes."""

import pandas as pd


def compute_pathway_metrics(
    assignments_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Join assignments to outcomes, group by (persona_tier, signal, product_code, experiment_version),
    compute treatment vs control means and uplift. significance_flag = True only if n_treatment + n_control >= min_n.
    Composite uplift_score from config weights.
    """
    from datetime import datetime, timezone

    exp = config.get("experiment", {})
    min_n = int(exp.get("min_n", 30))
    weights = exp.get("uplift_weights", {})
    w_conv = float(weights.get("conversion_uplift", 0.30))
    w_aua = float(weights.get("delta_aua_weight", 0.25))
    scale_aua = float(weights.get("delta_aua_uplift_scale", 10000.0))
    w_liq = float(weights.get("liquidity_uplift_weight", 0.20))
    w_ret = float(weights.get("retention_uplift_weight", 0.20))
    w_comp = float(weights.get("complaint_uplift_weight", -0.05))

    if assignments_df.empty or outcomes_df.empty:
        return pd.DataFrame()

    merged = outcomes_df.merge(
        assignments_df[["assignment_id", "persona_tier", "signal", "product_code", "assignment", "experiment_version"]],
        on="assignment_id",
        how="inner",
    )

    group_cols = ["persona_tier", "signal", "product_code", "experiment_version"]
    rows = []

    for key, grp in merged.groupby(group_cols):
        if isinstance(key, tuple) and len(key) == 4:
            persona_tier, signal, product_code, exp_ver = key
        else:
            persona_tier = grp["persona_tier"].iloc[0] if "persona_tier" in grp.columns else ""
            signal = grp["signal"].iloc[0] if "signal" in grp.columns else ""
            product_code = grp["product_code"].iloc[0] if "product_code" in grp.columns else ""
            exp_ver = grp["experiment_version"].iloc[0] if "experiment_version" in grp.columns else "v1"

        trt = grp[grp["assignment"] == "treatment"]
        ctl = grp[grp["assignment"] == "control"]

        n_treatment = len(trt)
        n_control = len(ctl)
        significance_flag = (n_treatment + n_control) >= min_n

        def mean_or_zero(s: pd.Series) -> float:
            return float(s.mean()) if len(s) > 0 else 0.0

        conv_uplift = mean_or_zero(trt["converted"]) - mean_or_zero(ctl["converted"])
        aua_uplift = mean_or_zero(trt["delta_aua"]) - mean_or_zero(ctl["delta_aua"])
        liq_uplift = mean_or_zero(trt["delta_liquidity_months"]) - mean_or_zero(ctl["delta_liquidity_months"])
        ret_uplift = mean_or_zero(trt["retained"]) - mean_or_zero(ctl["retained"])
        comp_uplift = mean_or_zero(trt["complaint"]) - mean_or_zero(ctl["complaint"])

        uplift_score = (
            w_conv * conv_uplift
            + w_aua * (aua_uplift / scale_aua)
            + w_liq * liq_uplift
            + w_ret * ret_uplift
            + w_comp * comp_uplift
        )

        rows.append({
            "persona_tier": persona_tier,
            "signal": signal,
            "product_code": product_code,
            "experiment_version": exp_ver,
            "n_treatment": n_treatment,
            "n_control": n_control,
            "conversion_uplift": conv_uplift,
            "delta_aua_uplift": aua_uplift,
            "liquidity_uplift": liq_uplift,
            "retention_uplift": ret_uplift,
            "complaint_uplift": comp_uplift,
            "uplift_score": uplift_score,
            "significance_flag": significance_flag,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
        })

    return pd.DataFrame(rows)
