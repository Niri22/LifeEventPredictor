"""
Growth Engine — Experiment oversight + Model reliability.

Three sections:
  A. Impact & Uplift (executive view)
  B. Experiment Health (treatment/control, significance, CI)
  C. Model Reliability (condensed precision/recall/F1, drift alerts)
"""

import pandas as pd
import streamlit as st

from ui.lib import (
    inject_ws_theme,
    get_default_macro,
    get_experiment_metrics,
    get_experiment_summary,
    build_ranked_experiment_table,
    experiment_persona_label,
    experiment_product_label,
    load_model_artifacts,
    get_model_reliability_table,
    load_data,
    MIDNIGHT,
    WS_GOLD,
    PERSONAS,
    DATA_PROCESSED,
    read_parquet,
    render_pulse_sidebar,
)

st.set_page_config(page_title="Growth Engine — W Pulse", page_icon="W", layout="wide")


def main():
    inject_ws_theme()

    if "macro" not in st.session_state:
        st.session_state.macro = get_default_macro()

    render_pulse_sidebar("growth")

    st.markdown('<div class="ws-main">', unsafe_allow_html=True)
    st.title("Growth Engine")
    st.caption("Experiment performance, pathway uplift, and model health — all in one view")

    metrics_df = get_experiment_metrics()

    # ==================================================================
    # SECTION A: Impact & Uplift (Executive View)
    # ==================================================================
    st.markdown("#### A. Impact & Uplift")

    if metrics_df.empty:
        st.info("No experiment metrics available. Run assignments and simulate outcomes via the demo script.")
    else:
        summary = get_experiment_summary(metrics_df)
        net_uplift = summary["net_uplift"]
        top_row = summary["top_row"]
        n_suppressed_sig = summary["n_suppressed_sig"]
        projected_aua = summary["projected_aua"]
        sorted_df = summary["sorted_df"]

        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            color = "color:#0d7d0d" if net_uplift >= 0 else "color:#c0392b"
            st.markdown(f'<p style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0">Net Uplift</p>'
                        f'<p style="font-size:2rem;font-weight:600;{color};margin:0">{net_uplift:+.2f}</p>', unsafe_allow_html=True)
            st.caption("Composite score, volume-weighted")
        with k2:
            if top_row is not None:
                st.markdown(f'<p style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0">Top Pathway</p>'
                            f'<p style="font-size:1.1rem;font-weight:600;margin:0">{experiment_persona_label(str(top_row["persona_tier"]))} + '
                            f'{experiment_product_label(str(top_row["product_code"]))}</p>', unsafe_allow_html=True)
                st.caption(f"Uplift: {float(top_row['uplift_score']) * 100:+.1f}%  |  AUA: ${float(top_row['delta_aua_uplift']):,.0f}")
            else:
                st.metric("Top Pathway", "—")
        with k3:
            if n_suppressed_sig > 0:
                st.markdown(f'<p style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0">Suppressed</p>'
                            f'<p style="font-size:2rem;font-weight:600;color:#c0392b;margin:0">{n_suppressed_sig}</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0">Suppressed</p>'
                            '<p style="font-size:1rem;margin:0"><span style="background:#E8F5E9;padding:4px 10px;border-radius:4px">None detected</span></p>',
                            unsafe_allow_html=True)
        with k4:
            aua_color = "color:#0d7d0d" if projected_aua >= 0 else "color:#c0392b"
            st.markdown(f'<p style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0">Projected AUA Impact</p>'
                        f'<p style="font-size:2rem;font-weight:600;{aua_color};margin:0">${projected_aua:,.0f}</p>', unsafe_allow_html=True)

        st.divider()

        # Ranked pathways table
        st.markdown("**Ranked Pathways**")
        ranked, styled = build_ranked_experiment_table(metrics_df)
        if styled is not None:
            st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()

        # Pathway deep dive
        if not ranked.empty:
            sel_idx = st.selectbox(
                "Deep dive into pathway",
                range(len(ranked)),
                format_func=lambda i: f"{ranked.iloc[i]['Persona']} · {ranked.iloc[i]['Product']} ({ranked.iloc[i]['Uplift %']:+.2f}%)",
                key="ge_pathway_select",
            )
            if 0 <= sel_idx < len(ranked):
                r = sorted_df.iloc[sel_idx]
                dd1, dd2, dd3 = st.columns(3)
                with dd1:
                    st.markdown("**Impact**")
                    _colored_metric("Conversion lift", float(r["conversion_uplift"]))
                    _colored_metric("AUA delta", float(r["delta_aua_uplift"]), fmt="${:,.0f}")
                    _colored_metric("Retention lift", float(r["retention_uplift"]))
                with dd2:
                    st.markdown("**Risk**")
                    _colored_metric("Liquidity impact", float(r["liquidity_uplift"]))
                    _colored_metric("Complaint lift", float(r["complaint_uplift"]), invert=True)
                with dd3:
                    st.markdown("**Statistical Confidence**")
                    st.metric("Treatment n", int(r["n_treatment"]))
                    st.metric("Control n", int(r["n_control"]))
                    sig = "Yes" if r["significance_flag"] else "No"
                    st.metric("Significance", sig)
                    # Approximate CI bar
                    u = float(r["uplift_score"])
                    ci_lo, ci_hi = u - 0.15, u + 0.15
                    st.caption(f"Uplift 95% CI (approx): [{ci_lo:+.2f}, {ci_hi:+.2f}]")
                    _render_ci_bar(ci_lo, u, ci_hi)

                # Action recommendation
                u_val, sig_flag = float(r["uplift_score"]), r["significance_flag"]
                if sig_flag and u_val > 0:
                    st.success("**Scale** — Increase prioritization weight. Strong positive uplift confirmed.")
                elif sig_flag and u_val < 0:
                    st.error("**Suppress** — Negative uplift confirmed. Do not scale this pathway.")
                else:
                    st.info("**Hold** — Insufficient evidence. Continue collecting data before acting.")

    st.divider()

    # ==================================================================
    # SECTION B: Experiment Health
    # ==================================================================
    st.markdown("#### B. Experiment Health")

    if metrics_df.empty:
        st.caption("No experiment data to display.")
    else:
        eh1, eh2, eh3 = st.columns(3)
        total_treatment = int(metrics_df["n_treatment"].sum())
        total_control = int(metrics_df["n_control"].sum())
        n_sig = int((metrics_df["significance_flag"] == True).sum())
        n_insig = len(metrics_df) - n_sig

        with eh1:
            st.markdown("**Volume**")
            st.metric("Treatment", f"{total_treatment:,}")
            st.metric("Control", f"{total_control:,}")
            st.metric("Total pathways", len(metrics_df))
        with eh2:
            st.markdown("**Significance**")
            st.metric("Significant pathways", n_sig)
            st.metric("Insignificant", n_insig)
            if len(metrics_df) > 0:
                st.caption(f"Significance rate: {n_sig / len(metrics_df):.0%}")
        with eh3:
            st.markdown("**Stability**")
            st.caption("Lift stability over time requires multiple experiment runs.")
            st.caption("Current status: **First run** — baseline established.")
            st.caption("Monitor across subsequent runs for trend validation.")

    st.divider()

    # ==================================================================
    # SECTION C: Model Reliability
    # ==================================================================
    st.markdown("#### C. Model Reliability")

    artifacts = load_model_artifacts()
    if not artifacts:
        st.warning("No model artifacts found. Train models first.")
    else:
        reliability_df = get_model_reliability_table(artifacts)

        def _drift_style(row):
            if row["Drift Alert"] == "Below target":
                return ["background-color: #FFEBEE;"] * len(row)
            return [""] * len(row)

        styled_rel = reliability_df.style.apply(_drift_style, axis=1).format(
            {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "AUC": "{:.3f}"},
            na_rep="—",
        )
        st.dataframe(styled_rel, use_container_width=True, hide_index=True)

        any_drift = (reliability_df["Drift Alert"] == "Below target").any()
        if any_drift:
            st.warning("⚠️ One or more models below precision target (0.80). Consider retraining on fresh data.")
        else:
            st.success("✓ All models above precision target.")

        # Feature shift check
        with st.expander("Feature Shift & Retraining", expanded=False):
            try:
                feat_df = read_parquet(DATA_PROCESSED / "features.parquet")
                latest_month = feat_df["month"].max()
                latest = feat_df[feat_df["month"] == latest_month]
                key_cols = ["spend_velocity_30d", "savings_rate_delta", "aua_current", "cc_spend_30d"]
                available = [c for c in key_cols if c in latest.columns]
                if available:
                    summary = latest[available].describe().loc[["mean", "std"]].round(2)
                    st.markdown("**Key feature statistics (latest month)**")
                    st.dataframe(summary, use_container_width=True)
                    st.caption("Compare against training distribution for drift detection. Significant shifts warrant retraining.")
                else:
                    st.caption("Feature columns not available for summary.")
            except Exception:
                st.caption("Could not load feature data for drift analysis.")

            st.markdown("**Retraining Status**")
            st.caption("Last trained: Synthetic data v1. Retrain on production data quarterly or when drift alerts trigger.")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _colored_metric(label: str, value: float, fmt: str = "{:+.2f}", invert: bool = False):
    """Display a metric with green/red coloring based on sign."""
    formatted = fmt.format(value)
    positive = value > 0
    if invert:
        positive = not positive
    color = "#0d7d0d" if positive else "#c0392b" if value != 0 else MIDNIGHT
    st.markdown(f'{label}: <span style="color:{color};font-weight:600">{formatted}</span>', unsafe_allow_html=True)


def _render_ci_bar(lo: float, point: float, hi: float):
    """Simple horizontal CI visualization."""
    range_total = max(abs(lo), abs(hi), 0.01) * 2
    left_pct = max(0, min(100, (lo - (-range_total / 2)) / range_total * 100))
    right_pct = max(0, min(100, (hi - (-range_total / 2)) / range_total * 100))
    mid_pct = max(0, min(100, (point - (-range_total / 2)) / range_total * 100))
    bar_color = "#0d7d0d" if point >= 0 else "#c0392b"
    st.markdown(
        f'<div style="position:relative;height:12px;background:#eee;border-radius:6px;margin:4px 0">'
        f'<div style="position:absolute;left:{left_pct:.0f}%;width:{right_pct - left_pct:.0f}%;height:100%;background:{bar_color}22;border-radius:6px"></div>'
        f'<div style="position:absolute;left:{mid_pct:.0f}%;width:4px;height:100%;background:{bar_color};border-radius:2px;transform:translateX(-2px)"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


main()
