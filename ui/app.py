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
    MIDNIGHT,
    WS_GOLD,
    GOV_TIER_ICONS,
    SIGNAL_LABELS,
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

    profiles, txns, features = load_data()
    model = load_model()

    # ------------------------------------------------------------------
    # Sidebar: W Pulse header, Filters, Scenario Planning (only)
    # ------------------------------------------------------------------
    st.sidebar.title("W Pulse")
    st.sidebar.caption("AI Growth Control Panel")
    st.sidebar.divider()

    with st.sidebar.expander("Filters", expanded=True):
        tier_filter = st.multiselect(
            "Persona Tier",
            options=[k for k in TIER_LABELS if k != "not_eligible"],
            format_func=lambda x: TIER_LABELS[x],
            default=["aspiring_affluent", "sticky_family_leader", "generation_nerd"],
        )
        confidence_min = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)

    with st.sidebar.expander("Scenario Planning", expanded=False):
        boc_rate = st.slider(
            "BoC Prime Rate (%)", 3.0, 8.0,
            float(st.session_state.macro.boc_prime_rate), 0.25, key="sb_boc",
        )
        vix_val = st.slider(
            "Market Volatility (VIX)", 10, 40,
            int(st.session_state.macro.vix), 1, key="sb_vix",
        )
        st.session_state.macro = MacroSnapshot(boc_prime_rate=boc_rate, vix=vix_val)

    macro = st.session_state.macro

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
            st.sidebar.warning("BoC rate > 6%: Retirement Accelerator suggestions hidden.")

    # Store hypotheses in session for Decision Console
    st.session_state["hypotheses"] = hypotheses
    st.session_state["filtered"] = filtered

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.markdown('<div class="ws-main">', unsafe_allow_html=True)
    st.title("Wealthsimple Pulse")
    st.caption("AI Growth Control Panel — Executive Command Layer")

    # ==================================================================
    # SECTION 1: System Health KPIs
    # ==================================================================
    st.markdown("#### System Health")

    decided_ids = set(st.session_state.decisions.keys())
    n_pending = len([h for h in filtered if h["user_id"] not in decided_ids])
    n_auto_approved = len([
        h for h in filtered
        if h.get("governance", {}).get("tier") == "green"
        and st.session_state.decisions.get(h["user_id"], {}).get("action") == "approved"
    ])
    n_suppressed = len([h for h in filtered if h.get("governance", {}).get("tier") == "red"])

    metrics_df = get_experiment_metrics()
    exp_summary = get_experiment_summary(metrics_df) if not metrics_df.empty else {}
    net_uplift = exp_summary.get("net_uplift", 0)
    projected_aua = exp_summary.get("projected_aua", 0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Active Signals", len(hypotheses))
    c2.metric("Pending Review", n_pending)
    c3.metric("Auto-Approved", n_auto_approved)
    c4.metric("Suppressed", n_suppressed)
    c5.metric("Net Uplift (30d)", f"{net_uplift:+.2f}")
    c6.metric("Projected AUA", f"${projected_aua:,.0f}")

    st.divider()

    # ==================================================================
    # SECTION 2: Strategic Levers
    # ==================================================================
    st.markdown("#### Strategic Levers")
    lev1, lev2, lev3, lev4 = st.columns(4)

    with lev1:
        rate_label = "High" if macro.rates_high else "Normal"
        vol_label = "Elevated" if macro.market_volatile else "Normal"
        st.markdown("**Macro Regime**")
        st.markdown(f"BoC: **{macro.boc_prime_rate:.2f}%** ({rate_label})")
        st.markdown(f"VIX: **{macro.vix:.0f}** ({vol_label})")

    with lev2:
        st.markdown("**Governance Thresholds**")
        st.markdown("🟢 Green > 0.90 confidence")
        st.markdown("🟡 Amber 0.70 – 0.90")
        st.markdown("🔴 Red < 0.70 or high-risk product")

    with lev3:
        try:
            cfg = load_config()
            clamp = cfg.get("experiment", {}).get("clamp_bounds", {})
            st.markdown("**Prioritization Weights**")
            st.markdown(f"Uplift clamp: [{clamp.get('uplift_weight_min', -0.25):+.0%}, {clamp.get('uplift_weight_max', 0.20):+.0%}]")
        except Exception:
            st.markdown("**Prioritization Weights**")
            st.markdown("Uplift clamp: [-25%, +20%]")

    with lev4:
        st.markdown("**Model Precision**")
        artifacts = load_model_artifacts()
        all_ok = True
        for persona, art in artifacts.items():
            prec = art.get("metrics", {}).get("precision", 0)
            label = experiment_persona_label(persona)
            if prec < 0.80:
                st.markdown(f"⚠️ {label}: **{prec:.2f}**")
                all_ok = False
            else:
                st.markdown(f"✓ {label}: {prec:.2f}")
        if all_ok and artifacts:
            st.caption("All models above 0.80 target.")

    st.divider()

    # ==================================================================
    # SECTION 3: Top Actions Required
    # ==================================================================
    st.markdown("#### Top Actions Required")

    red_cases = [h for h in filtered if h.get("governance", {}).get("tier") == "red"]
    amber_cases = [h for h in filtered if h.get("governance", {}).get("tier") == "amber"]

    alerts_rendered = 0

    # Red cases needing manual review
    if red_cases:
        liq_count = sum(1 for h in red_cases if h.get("signal") == "liquidity_warning")
        other_red = len(red_cases) - liq_count
        if liq_count:
            st.markdown(
                f'<div class="ws-alert ws-alert-red">🔴 <strong>{liq_count} Liquidity Watchdog case{"s" if liq_count != 1 else ""}</strong> require manual review</div>',
                unsafe_allow_html=True,
            )
            alerts_rendered += 1
        if other_red:
            st.markdown(
                f'<div class="ws-alert ws-alert-red">🔴 <strong>{other_red} high-risk signal{"s" if other_red != 1 else ""}</strong> flagged for 1-to-1 review</div>',
                unsafe_allow_html=True,
            )
            alerts_rendered += 1

    # Top boosted pathway from experiment metrics
    if exp_summary.get("top_row") is not None:
        top = exp_summary["top_row"]
        uplift_pct = float(top["uplift_score"]) * 100
        if uplift_pct > 0:
            st.markdown(
                f'<div class="ws-alert ws-alert-green">⬆ <strong>{experiment_persona_label(str(top["persona_tier"]))} + '
                f'{experiment_product_label(str(top["product_code"]))}</strong> showing <strong>+{uplift_pct:.1f}%</strong> uplift</div>',
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
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("Open Decision Console →", type="primary", use_container_width=True):
            st.switch_page("pages/1_decision_console.py")
    with col_nav2:
        if st.button("Open Growth Engine →", use_container_width=True):
            st.switch_page("pages/2_growth_engine.py")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
