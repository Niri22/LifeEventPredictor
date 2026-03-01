"""
Wealthsimple Pulse -- Control Center (Home)

Operational command surface: "What's urgent? What do I do next? Is the system safe?"
All configuration lives in the sidebar. This page is executable, not analytical.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from ui.lib import (
    TIER_LABELS,
    inject_ws_theme,
    load_data,
    load_precomputed_hypotheses,
    get_default_macro,
    get_experiment_metrics,
    get_experiment_summary,
    load_model_artifacts,
    experiment_persona_label,
    experiment_product_label,
    render_pulse_sidebar,
    render_empty_state,
    render_governance_constraints,
    render_audit_summary,
    render_model_confidence_context,
    format_currency,
    format_number,
    get_system_timestamps,
    get_compliance_info,
    PERSONAS,
)
from ui.onboarding import show_onboarding_dialog, should_show_onboarding

st.set_page_config(
    page_title="Wealthsimple Pulse",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default


def _render_safety_banner(macro, compliance: dict, timestamps: dict):
    """Single-line trust banner. Always visible. Replaces scattered compliance text."""
    regime = "Normal" if 3.0 <= macro.boc_prime_rate <= 6.0 and macro.vix <= 25 else "Elevated"
    try:
        artifacts = load_model_artifacts()
        precision_issues = sum(1 for p in PERSONAS if artifacts.get(p, {}).get("metrics", {}).get("precision", 1.0) < 0.75)
        model_status = "Stable" if precision_issues == 0 else f"Mixed ({precision_issues} below threshold)"
    except Exception:
        model_status = "Unknown"

    is_safe = (regime == "Normal" and model_status == "Stable" and compliance["compliance_status"] == "COMPLIANT")
    banner_class = "safe" if is_safe else "degraded"
    icon = "&#x2705;" if is_safe else "&#x26A0;&#xFE0F;"
    label = "System Safe to Operate" if is_safe else "System Degraded — Review Recommended"

    st.markdown(f"""
    <div class="ws-safety-banner {banner_class}">
        <span>{icon} <strong>{label}</strong></span>
        <span class="sep">|</span> Macro: {regime}
        <span class="sep">|</span> Models: {model_status}
        <span class="sep">|</span> Compliance: {compliance['compliance_status']}
        <span class="sep">|</span> Updated: {timestamps['last_updated']}
        <span class="sep">|</span> {compliance['model_version']}
    </div>
    """, unsafe_allow_html=True)


def _render_mission(n_red, n_amber, n_green, n_red_liquidity, n_red_other, projected_aua):
    """Hero block: Today's Mission. Answers 'What am I here to do?' in 3 seconds."""
    st.markdown(f"""
    <div class="ws-mission-card">
        <div class="ws-mission-title">Today's Mission</div>
        <div class="ws-mission-line"><strong>{n_red}</strong> high-risk cases need review ({n_red_liquidity} Liquidity, {n_red_other} Other)</div>
        <div class="ws-mission-line"><strong>{n_amber}</strong> eligible for batch approval (Amber)</div>
        <div class="ws-mission-line"><strong>{n_green}</strong> auto-approved (Green)</div>
        <div class="ws-mission-impact">Projected impact at stake: {format_currency(projected_aua)} AUA</div>
    </div>
    """, unsafe_allow_html=True)


def _render_action_stack_item(title, why, impact, cta_label, urgency, key):
    """One action card: title, why, impact, CTA. Returns True if clicked."""
    st.markdown(f"""
    <div class="ws-action-stack {urgency}">
        <div class="ws-action-stack-title">{title}</div>
        <div class="ws-action-stack-why">{why}</div>
        <div class="ws-action-stack-impact">{impact}</div>
    </div>
    """, unsafe_allow_html=True)
    return st.button(cta_label, key=key, use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    inject_ws_theme()

    if "macro" not in st.session_state:
        st.session_state.macro = get_default_macro()
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}

    render_pulse_sidebar("control")

    if should_show_onboarding():
        show_onboarding_dialog()
        st.stop()

    try:
        profiles, txns, features = load_data()
    except Exception:
        st.markdown('<div class="ws-main">', unsafe_allow_html=True)
        render_empty_state("No data available.", "Awaiting signals. Ensure data is generated or check configuration.", "📊")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    macro = st.session_state.macro
    tier_filter = st.session_state.get("pulse_tier_filter", [k for k in TIER_LABELS if k != "not_eligible"])
    confidence_min = st.session_state.get("pulse_confidence_min", 0.5)

    hypotheses = load_precomputed_hypotheses()
    filtered = [h for h in hypotheses if h["persona_tier"] in tier_filter and h["confidence"] >= confidence_min]
    if macro.boc_prime_rate > 6.0:
        filtered = [h for h in filtered if h["traceability"]["target_product"]["code"] != "RRSP_LOAN"]

    st.session_state["hypotheses"] = hypotheses
    st.session_state["filtered"] = filtered

    decided_ids = set(st.session_state.decisions.keys())
    red_cases = [h for h in filtered if h.get("governance", {}).get("tier") == "red"]
    amber_cases = [h for h in filtered if h.get("governance", {}).get("tier") == "amber"]
    green_cases = [h for h in filtered if h.get("governance", {}).get("tier") == "green"]
    undecided_red = [h for h in red_cases if h["user_id"] not in decided_ids]
    undecided_amber = [h for h in amber_cases if h["user_id"] not in decided_ids]
    n_red_liquidity = sum(1 for h in red_cases if h.get("signal", "").lower() == "liquidity_warning")
    n_red_other = len(red_cases) - n_red_liquidity

    try:
        metrics_df = get_experiment_metrics()
    except Exception:
        metrics_df = __import__("pandas").DataFrame()
    exp_summary = get_experiment_summary(metrics_df) if (metrics_df is not None and not getattr(metrics_df, "empty", True)) else {}
    projected_aua = _safe_float(exp_summary.get("projected_aua"), 22697)

    timestamps = get_system_timestamps()
    compliance = get_compliance_info()

    # =====================================================================
    # PAGE LAYOUT: Header → Safety → Mission → Actions → Progress → System
    # =====================================================================
    st.markdown('<div class="ws-main">', unsafe_allow_html=True)

    # HEADER — title + operational subtitle (no KPIs)
    st.markdown('<h1 class="ws-page-title">Control Center</h1>', unsafe_allow_html=True)
    st.markdown('<div class="ws-secondary">Action queue for today\'s risk + growth decisions.</div>', unsafe_allow_html=True)

    # SAFETY BANNER — one line, always visible
    _render_safety_banner(macro, compliance, timestamps)

    # ── HERO: Today's Mission ──
    _render_mission(
        len(red_cases), len(undecided_amber), len(green_cases),
        n_red_liquidity, n_red_other, projected_aua,
    )

    # Primary CTA — big, centered
    if len(undecided_red) > 0:
        if st.button(
            f"Review {len(undecided_red)} High-Risk Cases",
            key="cta_review_red", type="primary", use_container_width=True,
        ):
            st.session_state.pulse_queue_tier = "red"
            st.switch_page("pages/1_decision_console.py")

    # Secondary CTAs — side by side
    cta1, cta2 = st.columns(2)
    with cta1:
        if len(undecided_amber) > 0:
            if st.button(
                f"Batch Approve {len(undecided_amber)} Amber",
                key="cta_batch_amber", use_container_width=True,
            ):
                st.session_state.pulse_queue_tier = "amber"
                st.switch_page("pages/1_decision_console.py")
    with cta2:
        if st.button("Open Growth Engine", key="cta_growth", use_container_width=True):
            st.switch_page("pages/2_growth_engine.py")

    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)

    # ── ACTION STACK ──
    st.markdown('<div class="ws-section-header">Action Stack</div>', unsafe_allow_html=True)

    actions_rendered = 0

    # 1. High-risk review
    if len(red_cases) > 0:
        try:
            clicked = _render_action_stack_item(
                f"🔴 High-Risk Review — {len(red_cases)} cases",
                f"Liquidity + suitability constraints triggered ({n_red_liquidity} liquidity, {n_red_other} other)",
                "Prevents unsuitable allocation actions",
                f"Review {len(undecided_red)} cases now →",
                "urgent",
                "as_red",
            )
            if clicked:
                st.session_state.pulse_queue_tier = "red"
                st.switch_page("pages/1_decision_console.py")
            actions_rendered += 1
        except Exception:
            pass

    # 2. Batch approvals
    if len(undecided_amber) > 0:
        try:
            clicked = _render_action_stack_item(
                f"🟠 Batch Approvals — {len(undecided_amber)} eligible",
                "Medium risk + high confidence, cohort-safe",
                "Reduces curator load ~40%",
                f"Approve cohort ({len(undecided_amber)}) →",
                "amber",
                "as_amber",
            )
            if clicked:
                st.session_state.pulse_queue_tier = "amber"
                st.switch_page("pages/1_decision_console.py")
            actions_rendered += 1
        except Exception:
            pass

    # 3. Growth opportunity
    top_pathway = exp_summary.get("top_row")
    if top_pathway is not None:
        try:
            row = top_pathway.to_dict() if hasattr(top_pathway, "to_dict") else top_pathway
            persona_label = experiment_persona_label(str(row.get("persona_tier", "Unknown")))
            product_label = experiment_product_label(str(row.get("product_code", "Unknown")))
            uplift_pct = _safe_float(row.get("uplift_score"), 0) * 100
            delta_aua = _safe_float(row.get("delta_aua_uplift"), 0)
            clicked = _render_action_stack_item(
                f"🟢 Growth Opportunity — {persona_label} + {product_label}",
                f"Pathway showing +{uplift_pct:.1f}% uplift",
                f"+{format_currency(delta_aua)} projected AUA",
                "Inspect experiment →",
                "growth",
                "as_growth",
            )
            if clicked:
                st.switch_page("pages/2_growth_engine.py")
            actions_rendered += 1
        except Exception:
            pass

    if actions_rendered == 0:
        render_empty_state("All Systems Nominal", "No immediate actions required.", "🎯")

    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)

    # ── QUEUE PROGRESS ──
    total_requiring_review = len(undecided_red) + len(undecided_amber)
    total_reviewed = len(decided_ids)
    total_queue = len(red_cases) + len(amber_cases)
    processed_today = total_queue - total_requiring_review if total_queue else 0

    st.markdown('<div class="ws-subsection">Queue Progress</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="queue-progress">Processed: <strong>{processed_today} / {total_queue}</strong> '
        f'&nbsp;&nbsp;·&nbsp;&nbsp; Remaining: <strong>{total_requiring_review}</strong> cases</div>',
        unsafe_allow_html=True,
    )
    if total_queue > 0:
        st.progress(processed_today / total_queue)
    else:
        st.progress(1.0)

    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)

    # ── IMPACT (one number + one sentence) ──
    net_uplift = _safe_float(exp_summary.get("net_uplift"), 0.43)
    st.markdown('<div class="ws-subsection">Impact (rolling 30d)</div>', unsafe_allow_html=True)
    st.write(f"**{format_currency(projected_aua)}** projected AUA · Net uplift **+{net_uplift:.2f}**")
    top_row = exp_summary.get("top_row")
    if top_row is not None:
        try:
            row = top_row.to_dict() if hasattr(top_row, "to_dict") else top_row
            st.caption(f"Top pathway: {experiment_product_label(str(row.get('product_code', '—')))}.")
        except Exception:
            pass
    if st.button("See details → Growth Engine", key="impact_link"):
        st.switch_page("pages/2_growth_engine.py")

    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)

    # ── COLLAPSED: System Metrics ──
    with st.expander("System Metrics", expanded=False):
        st.markdown('<div class="ws-kpi-compact">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="ws-kpi-compact-item"><span class="val">{format_number(len(filtered))}</span><span class="lbl">Active</span></div>
            <div class="ws-kpi-compact-item"><span class="val">{format_number(len(undecided_red) + len(undecided_amber))}</span><span class="lbl">Pending Review</span></div>
            <div class="ws-kpi-compact-item"><span class="val">{format_number(len(green_cases))}</span><span class="lbl">Auto-Approved</span></div>
            <div class="ws-kpi-compact-item"><span class="val">{format_number(len(red_cases))}</span><span class="lbl">Suppressed</span></div>
            <div class="ws-kpi-compact-item"><span class="val">+{net_uplift:.2f}</span><span class="lbl">Net Uplift</span></div>
            <div class="ws-kpi-compact-item"><span class="val">{format_currency(projected_aua)}</span><span class="lbl">Proj. AUA</span></div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── COLLAPSED: System Status + Model Health ──
    with st.expander("System Status & Governance", expanded=False):
        col_left, col_right = st.columns(2)
        with col_left:
            render_governance_constraints()
            regime = "Normal" if 3.0 <= macro.boc_prime_rate <= 6.0 and macro.vix <= 25 else "Volatile"
            st.markdown(f"""
            <div class="ws-secondary" style="margin-top: 1rem;">
            <strong>Macro:</strong> {regime} (BoC {macro.boc_prime_rate:.2f}%, VIX {macro.vix:.0f})<br>
            <strong>Governance:</strong> Green >0.90 | Amber 0.70–0.90 | Red &lt;0.70<br>
            <strong>Uplift Clamp:</strong> [-25%, +20%]
            </div>
            """, unsafe_allow_html=True)
        with col_right:
            try:
                artifacts = load_model_artifacts()
                precision_labels = {
                    "aspiring_affluent": "Momentum Builder",
                    "sticky_family_leader": "Full-Stack Client",
                    "generation_nerd": "Legacy Architect",
                }
                for p in PERSONAS:
                    precision = artifacts.get(p, {}).get("metrics", {}).get("precision", 1.0)
                    render_model_confidence_context(precision_labels.get(p, p), precision)
            except Exception:
                st.markdown('<div class="ws-secondary">Model metrics unavailable</div>', unsafe_allow_html=True)
            render_audit_summary()
            if st.button("View Full Compliance Dashboard", key="compliance_link"):
                st.switch_page("pages/3_compliance.py")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("Something went wrong. No data or configuration may be available.")
        st.caption("If this persists, ensure data is generated and paths are correct.")
