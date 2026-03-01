"""
Wealthsimple Pulse -- Control Center (Home)

Executive command layer: System Health KPIs, Strategic Levers, Top Actions Required.
All case-level review lives in Decision Console; experiment/model oversight in Growth Engine.
"""

import sys
from pathlib import Path

# Ensure project root is on path when running as streamlit run ui/app.py (e.g. on Streamlit Cloud)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from datetime import datetime, timezone

import streamlit as st

from src.api.feedback import get_feedback_stats
from src.api.macro_agent import MacroSnapshot

from ui.lib import (
    TIER_LABELS,
    GOV_TIER_ICONS,
    inject_ws_theme,
    load_data,
    load_precomputed_hypotheses,
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

    # Sidebar first so Control Center / Decision Console / Growth Engine always load when clicked
    from ui.lib import render_pulse_sidebar, render_kpi_card, render_action_card, render_empty_state, format_currency, format_number, get_last_updated, ITEM_TERMINOLOGY
    render_pulse_sidebar("control")

    # If onboarding not completed, show tour in main area only; rest of page stays empty
    if should_show_onboarding():
        show_onboarding_dialog()
        st.stop()

    profiles, txns, features = load_data()
    macro = st.session_state.macro
    tier_filter = st.session_state.get("pulse_tier_filter", [k for k in TIER_LABELS if k != "not_eligible"])
    confidence_min = st.session_state.get("pulse_confidence_min", 0.5)

    # Pre-computed prototype mode: load static hypotheses (no model inference)
    hypotheses = load_precomputed_hypotheses()
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
        and h["user_id"] not in decided_ids
    ])
    n_suppressed = len([h for h in filtered if h.get("governance", {}).get("tier") == "red"])
    red_cases = [h for h in filtered if h.get("governance", {}).get("tier") == "red"]
    amber_cases = [h for h in filtered if h.get("governance", {}).get("tier") == "amber"]
    undecided_amber = [h for h in amber_cases if h["user_id"] not in decided_ids]

    metrics_df = get_experiment_metrics()
    exp_summary = get_experiment_summary(metrics_df) if not metrics_df.empty else {}

    # ---------------------------------------------------------------------------
    # Main content
    # ---------------------------------------------------------------------------
    st.markdown('<div class="ws-main">', unsafe_allow_html=True)

    # Header with executive summary and last updated - answers "What needs attention now?" in 5 seconds
    col_title, col_updated = st.columns([3, 1])
    with col_title:
        st.markdown('<h1 class="ws-heading">Control Center</h1>', unsafe_allow_html=True)
    with col_updated:
        st.markdown(f"**Last Updated:** {get_last_updated()}")

    # Executive summary line
    total_monitored = len(hypotheses)
    total_need_decision = n_pending + n_suppressed
    total_batch_eligible = len(undecided_amber)
    
    st.markdown(f"""
    **{total_monitored} {ITEM_TERMINOLOGY.lower()} monitored • {total_need_decision} require human review • {total_batch_eligible} eligible for batch approval**
    """)
    st.markdown("---")

    # System Health KPIs - grouped Operational vs Impact
    st.markdown("### System Health")
    
    st.markdown("**Operational**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Active", format_number(len(filtered)))
    with col2:
        render_kpi_card("Pending Review", format_number(n_pending), 
                       f"+{max(0, n_pending-20)}" if n_pending > 20 else None, "neutral")
    with col3:
        render_kpi_card("Auto-Approved", format_number(n_auto_approved))
    with col4:
        render_kpi_card("Suppressed", format_number(n_suppressed),
                       f"+{n_suppressed}" if n_suppressed > 0 else None, "negative" if n_suppressed > 0 else "neutral")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Impact**")
    col5, col6 = st.columns(2)
    with col5:
        net_uplift = exp_summary.get("net_uplift", 0.43)
        render_kpi_card("Net Uplift (30d)", f"+{net_uplift:.2f}", "composite score", "positive")
    with col6:
        projected_aua = exp_summary.get("projected_aua", 22697)
        render_kpi_card("Projected AUA Impact", format_currency(projected_aua))

    st.markdown("---")

    # Top Actions Required (HERO SECTION - most important)
    st.markdown("### Top Actions Required")
    
    actions_rendered = 0
    
    # High-risk cases (most urgent)
    if n_suppressed > 0:
        liquidity_cases = sum(1 for h in red_cases 
                            if h.get("signal", "").lower() == "liquidity_warning")
        other_cases = n_suppressed - liquidity_cases
        
        subtitle = f"{liquidity_cases} Liquidity | {other_cases} Other" if liquidity_cases > 0 else f"{n_suppressed} High-Risk {ITEM_TERMINOLOGY}"
        
        if render_action_card(
            f"🚨 {n_suppressed} High-Risk {ITEM_TERMINOLOGY} Require Manual Review",
            subtitle,
            f"Review {n_suppressed} High-Risk {ITEM_TERMINOLOGY}",
            "urgent",
            "action_high_risk"
        ):
            st.switch_page("pages/1_decision_console.py")
        actions_rendered += 1
    
    # Growth opportunity
    top_pathway = exp_summary.get("top_row")
    if top_pathway and actions_rendered < 3:
        persona_label = experiment_persona_label(str(top_pathway.get('persona_tier', 'Unknown')))
        product_label = experiment_product_label(str(top_pathway.get('product_code', 'Unknown')))
        uplift_pct = float(top_pathway.get("uplift_score", 0)) * 100
        projected_aua_growth = float(top_pathway.get("delta_aua_uplift", 0))
        
        if render_action_card(
            f"⬆ {persona_label} + {product_label} showing +{uplift_pct:.1f}% uplift",
            f"Projected {format_currency(projected_aua_growth)} AUA impact",
            "Analyze Pathway Performance",
            "growth",
            "action_growth"
        ):
            st.switch_page("pages/2_growth_engine.py")
        actions_rendered += 1
    
    # Batch approval opportunity
    if len(undecided_amber) > 0 and actions_rendered < 3:
        if render_action_card(
            f"✅ {len(undecided_amber)} {ITEM_TERMINOLOGY} Eligible for Batch Approval",
            f"Amber tier cases ready for automated processing",
            f"Batch Approve {len(undecided_amber)} {ITEM_TERMINOLOGY}",
            "normal",
            "action_batch"
        ):
            st.switch_page("pages/1_decision_console.py")
        actions_rendered += 1

    # Empty state if no actions
    if actions_rendered == 0:
        render_empty_state(
            "All Systems Nominal",
            "No immediate actions required. All cases are within normal parameters.",
            "🎯"
        )

    st.markdown("---")

    # Strategic Levers (compressed, not dominant)
    st.markdown("### Strategic Levers")
    
    # Single compact summary instead of 4 columns
    regime = "Normal" if 3.0 <= macro.boc_prime_rate <= 6.0 and macro.vix <= 25 else "Volatile"
    
    try:
        artifacts = load_model_artifacts()
        precision_issues = []
        precision_labels = {"aspiring_affluent": "MB", "sticky_family_leader": "FSC", "generation_nerd": "LA"}
        
        for p in ["aspiring_affluent", "sticky_family_leader", "generation_nerd"]:
            precision = artifacts.get(p, {}).get("metrics", {}).get("precision", 1.0)
            label = precision_labels.get(p, p[:2].upper())
            if precision < 0.75:
                precision_issues.append(f"⚠ {label} {precision:.2f}")
            else:
                precision_issues.append(f"✓ {label} {precision:.2f}")
        
        precision_summary = " | ".join(precision_issues)
    except:
        precision_summary = "Unknown"

    st.markdown(f"""
    **Macro:** {regime} (BoC {macro.boc_prime_rate:.2f}%, VIX {macro.vix})  
    **Governance:** Green >0.90 | Amber 0.70–0.90 | Red <0.70  
    **Uplift Clamp:** [-25%, +20%]  
    **Model Precision:** {precision_summary}
    """)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
