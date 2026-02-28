"""
Wealthsimple Pulse -- Control Center (Home)

Executive command layer: System Health KPIs, Strategic Levers, Top Actions Required.
All case-level review lives in Decision Console; experiment/model oversight in Growth Engine.
"""

from datetime import datetime, timezone

import streamlit as st

from src.api.feedback import get_feedback_stats
from src.api.macro_agent import MacroSnapshot

from ui.lib import (
    TIER_LABELS,
    GOV_TIER_ICONS,
    inject_ws_theme,
    load_data,
    load_model,
    generate_hypotheses,
    get_default_macro,
    get_experiment_metrics,
    get_experiment_summary,
    apply_experiment_reweight,
    load_model_artifacts,
    experiment_persona_label,
    experiment_product_label,
    load_config,
)
from ui.onboarding import show_onboarding_dialog, should_show_onboarding

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wealthsimple Pulse",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    inject_ws_theme()

    if "macro" not in st.session_state:
        st.session_state.macro = get_default_macro()
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}

    # First-time onboarding: show tour modal when not yet completed
    if should_show_onboarding():
        show_onboarding_dialog()

    profiles, txns, features = load_data()
    model = load_model()

    # ------------------------------------------------------------------
    # Sidebar: command-console nav + context + filters + utilities
    # ------------------------------------------------------------------
    from ui.lib import render_pulse_sidebar
    render_pulse_sidebar("control")
    macro = st.session_state.macro
    tier_filter = st.session_state.get("pulse_tier_filter", [k for k in TIER_LABELS if k != "not_eligible"])
    confidence_min = st.session_state.get("pulse_confidence_min", 0.5)

    # Generate hypotheses and apply filters
    hypotheses = generate_hypotheses(features, profiles, model, macro)
    filtered = [
        h for h in hypotheses
        if h["persona_tier"] in tier_filter and h["confidence"] >= confidence_min
    ]
    if macro.boc_prime_rate > 6.0:
        before_len = len(filtered)
        filtered = [h for h in filtered if h["traceability"]["target_product"]["code"] != "RRSP_LOAN"]
        if len(filtered) < before_len:
            st.warning("BoC rate > 6%: Retirement Accelerator suggestions hidden.")

    # Store hypotheses in session for Decision Console
    st.session_state["hypotheses"] = hypotheses
    st.session_state["filtered"] = filtered

    decided_ids = set(st.session_state.decisions.keys())
    n_pending = len([h for h in filtered if h["user_id"] not in decided_ids])
    n_auto_approved = len([
        h for h in filtered
        if h.get("governance", {}).get("tier") == "green"
        and st.session_state.decisions.get(h["user_id"], {}).get("action") == "approved"
    ])
    n_suppressed = len([h for h in filtered if h.get("governance", {}).get("tier") == "red"])
    red_cases = [h for h in filtered if h.get("governance", {}).get("tier") == "red"]
    amber_cases = [h for h in filtered if h.get("governance", {}).get("tier") == "amber"]
    undecided_amber = [h for h in amber_cases if h["user_id"] not in decided_ids]
    n_batch_eligible = len(undecided_amber)

    metrics_df = get_experiment_metrics()
    exp_summary = get_experiment_summary(metrics_df) if not metrics_df.empty else {}
    net_uplift = exp_summary.get("net_uplift", 0)
    projected_aua = exp_summary.get("projected_aua", 0)
    n_high_risk = len(red_cases)

    # ------------------------------------------------------------------
    # Header + executive summary + last updated
    # ------------------------------------------------------------------
    st.markdown('<div class="ws-main">', unsafe_allow_html=True)
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.title("Control Center")
        st.caption(
            f"{len(hypotheses)} signals monitored · "
            f"{n_pending} require human review · "
            f"{n_batch_eligible} eligible for batch approval"
        )
    with head_col2:
        st.caption("Updated 2m ago")
    st.markdown("<br>", unsafe_allow_html=True)

    # ==================================================================
    # System Health: Operational | Impact (spacing between clusters)
    # ==================================================================
    st.markdown("#### System Health")
    op1, op2, op3, op4 = st.columns(4)
    op1.metric("Active", len(hypotheses))
    op2.metric("Pending", n_pending)
    op3.metric("Auto", n_auto_approved)
    op4.metric("Suppressed", n_suppressed)
    st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)
    imp1, imp2 = st.columns(2)
    with imp1:
        st.metric("Net Uplift (30d)", f"{net_uplift:+.2f}")
        st.caption("composite score")
    with imp2:
        st.metric("Projected AUA", f"${projected_aua:,.0f}")

    st.divider()

    # ==================================================================
    # Strategic Levers (compact)
    # ==================================================================
    st.markdown("#### Strategic Levers")
    rate_label = "High" if macro.rates_high else "Normal"
    vol_label = "Elevated" if macro.market_volatile else "Normal"
    try:
        cfg = load_config()
        clamp = cfg.get("experiment", {}).get("clamp_bounds", {})
        clamp_str = f"[{clamp.get('uplift_weight_min', -0.25):+.0%}, {clamp.get('uplift_weight_max', 0.20):+.0%}]"
    except Exception:
        clamp_str = "[-25%, +20%]"
    artifacts = load_model_artifacts()
    precision_target = 0.80
    persona_short = {"aspiring_affluent": "MB", "sticky_family_leader": "FSC", "generation_nerd": "LA"}
    model_parts = []
    for persona, art in (artifacts or {}).items():
        prec = art.get("metrics", {}).get("precision", 0)
        short = persona_short.get(persona, persona[:2].upper())
        if prec < precision_target:
            model_parts.append(f"⚠ {short} {prec:.2f}")
        else:
            model_parts.append(f"✓ {short} {prec:.2f}")
    model_line = " | ".join(model_parts) if model_parts else "—"
    below_target = any(
        (artifacts or {}).get(p, {}).get("metrics", {}).get("precision", 1) < precision_target
        for p in ("aspiring_affluent", "sticky_family_leader", "generation_nerd")
    )
    precision_note = "Below target precision (0.80)" if below_target else "Within target (0.80)"
    st.markdown(
        f"Macro: {rate_label} (BoC {macro.boc_prime_rate:.2f}%, VIX {macro.vix:.0f}) · "
        f"Governance: Green >0.90 | Amber 0.70–0.90 | Red <0.70 · "
        f"Uplift Clamp: {clamp_str}"
    )
    st.markdown(f"Model precision: {precision_note} — {model_line}")
    st.divider()

    # ==================================================================
    # Top Actions Required
    # ==================================================================
    st.markdown("#### Top Actions Required")

    alerts_rendered = 0

    # Merged high-risk alert
    if red_cases:
        liq_count = sum(1 for h in red_cases if h.get("signal") == "liquidity_warning")
        other_red = n_high_risk - liq_count
        subline = f"{liq_count} Liquidity | {other_red} Other" if (liq_count and other_red) else (f"{liq_count} Liquidity" if liq_count else f"{other_red} Other")
        st.markdown(
            f'<div class="ws-alert ws-alert-red">🔴 <strong>{n_high_risk} High-Risk Cases Require Review</strong><br><span style="font-size:0.9em;opacity:0.9;">{subline}</span></div>',
            unsafe_allow_html=True,
        )
        alerts_rendered += 1

    # Top boosted pathway with AUA context
    if exp_summary.get("top_row") is not None:
        top = exp_summary["top_row"]
        uplift_pct = float(top["uplift_score"]) * 100
        top_aua = float(top.get("delta_aua_uplift", 0))
        if uplift_pct > 0:
            st.markdown(
                f'<div class="ws-alert ws-alert-green">↑ <strong>{experiment_persona_label(str(top["persona_tier"]))} + '
                f'{experiment_product_label(str(top["product_code"]))}</strong><br>'
                f'+{uplift_pct:.1f}% uplift · <strong>Projected +${top_aua:,.0f} AUA</strong></div>',
                unsafe_allow_html=True,
            )
            alerts_rendered += 1

    # Suppressed pathways
    n_supp = exp_summary.get("n_suppressed_sig", 0)
    if n_supp:
        st.markdown(
            f'<div class="ws-alert ws-alert-amber">⚠ <strong>{n_supp} pathway{"s" if n_supp != 1 else ""}</strong> suppressed due to negative uplift / macro volatility</div>',
            unsafe_allow_html=True,
        )
        alerts_rendered += 1

    # Amber batch opportunity
    if amber_cases:
        undecided_amber = [h for h in amber_cases if h["user_id"] not in decided_ids]
        if undecided_amber:
            st.markdown(
                f'<div class="ws-alert ws-alert-amber">🟡 <strong>{len(undecided_amber)} Amber case{"s" if len(undecided_amber) != 1 else ""}</strong> eligible for batch approval</div>',
                unsafe_allow_html=True,
            )
            alerts_rendered += 1

    # Safety brake triggers
    for h in filtered[:10]:
        _, safety = apply_experiment_reweight(h)
        if safety:
            for sa in safety:
                st.markdown(
                    f'<div class="ws-alert ws-alert-red">🛑 Safety brake: <strong>{sa.get("reason", "Unknown")}</strong> — {sa.get("metric", "")}</div>',
                    unsafe_allow_html=True,
                )
                alerts_rendered += 1
                break
        if alerts_rendered >= 6:
            break

    if alerts_rendered == 0:
        st.markdown(
            '<div class="ws-alert ws-alert-green">✓ No urgent actions. System operating within normal parameters.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("", unsafe_allow_html=True)
    cta_decision = f"Review {n_high_risk} High-Risk Cases →" if n_high_risk else "Decision Console →"
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button(cta_decision, type="primary", use_container_width=True):
            st.switch_page("pages/1_decision_console.py")
    with col_nav2:
        if st.button("Analyze Pathway Performance →", use_container_width=True):
            st.switch_page("pages/2_growth_engine.py")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
