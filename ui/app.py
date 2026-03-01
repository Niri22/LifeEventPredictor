"""
Wealthsimple Pulse -- Control Center (Home)

Operator control surface. Layout:
  Header → Status strip → Command bar → Action Stack + Progress → Impact → Collapsed system
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


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default


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
        render_empty_state("No data available.", "Awaiting signals.", "📊")
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
    net_uplift = _safe_float(exp_summary.get("net_uplift"), 0.43)
    timestamps = get_system_timestamps()
    compliance = get_compliance_info()

    # =====================================================================
    st.markdown('<div class="ws-main">', unsafe_allow_html=True)

    # ── HEADER ──
    st.markdown('<h1 class="ws-page-title">Control Center</h1>', unsafe_allow_html=True)
    st.markdown('<div class="ws-secondary" style="margin-bottom:0.25rem;padding-top:5px;padding-bottom:5px;">Action queue for today\'s risk + growth decisions.</div>', unsafe_allow_html=True)

    # ── STATUS STRIP — neutral when safe, prominent only when degraded ──
    regime = "Normal" if 3.0 <= macro.boc_prime_rate <= 6.0 and macro.vix <= 25 else "Elevated"
    try:
        artifacts = load_model_artifacts()
        precision_issues = sum(1 for p in PERSONAS if artifacts.get(p, {}).get("metrics", {}).get("precision", 1.0) < 0.75)
        model_status = "Stable" if precision_issues == 0 else f"Mixed ({precision_issues} below)"
    except Exception:
        model_status = "Unknown"

    is_safe = (regime == "Normal" and model_status == "Stable" and compliance["compliance_status"] == "COMPLIANT")
    strip_class = "ws-status-strip" if is_safe else "ws-status-strip degraded"
    st.markdown(f"""
    <div class="{strip_class}">
        Macro: {regime} <span class="sep">·</span>
        Models: {model_status} <span class="sep">·</span>
        {compliance['compliance_status']} <span class="sep">·</span>
        Updated {timestamps['last_updated']} <span class="sep">·</span>
        {compliance['model_version']}
    </div>
    """, unsafe_allow_html=True)

    # ── SUMMARY LINE — counts only, no buttons ──
    st.markdown(
        f'<div style="font-size:0.85rem;color:#64748b;margin-bottom:0.5rem;padding-top:5px;padding-bottom:5px;">'
        f'<strong style="color:#0f172a;">{len(red_cases)}</strong> High-Risk '
        f'<span style="color:#cbd5e1;">&nbsp;·&nbsp;</span>'
        f'<strong style="color:#0f172a;">{len(undecided_amber)}</strong> Amber '
        f'<span style="color:#cbd5e1;">&nbsp;·&nbsp;</span>'
        f'<strong style="color:#0f172a;">{len(green_cases)}</strong> Green '
        f'<span style="color:#cbd5e1;">&nbsp;·&nbsp;</span>'
        f'{format_currency(projected_aua)} at stake</div>',
        unsafe_allow_html=True,
    )

    # ── ACTION STACK — each card is one st.columns row: [text | button] ──
    st.markdown('<div class="ws-section-header">Action Stack</div>', unsafe_allow_html=True)

    actions_rendered = 0

    # 1. High-risk
    if len(red_cases) > 0:
        try:
            c_text, c_btn = st.columns([4, 1])
            with c_text:
                st.markdown(
                    f'<div class="ws-action-card-highrisk" style="padding:0.6rem 0.9rem;">'
                    f'<div style="font-weight:600;font-size:0.9rem;">High-Risk Review — {len(red_cases)} cases</div>'
                    f'<div style="font-size:0.8rem;color:#64748b;">Liquidity + suitability ({n_red_liquidity} liquidity, {n_red_other} other) · Prevents unsuitable allocation</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with c_btn:
                st.markdown('<div class="btn-urgent">', unsafe_allow_html=True)
                if st.button(f"Review ({len(undecided_red)})", key="as_red"):
                    st.session_state.pulse_queue_tier = "red"
                    st.switch_page("pages/1_decision_console.py")
                st.markdown("</div>", unsafe_allow_html=True)
            actions_rendered += 1
        except Exception:
            pass

    # 2. Batch approvals
    if len(undecided_amber) > 0:
        try:
            c_text, c_btn = st.columns([4, 1])
            with c_text:
                st.markdown(
                    f'<div class="ws-action-card-batch" style="padding:0.6rem 0.9rem;">'
                    f'<div style="font-weight:600;font-size:0.9rem;">Batch Approvals — {len(undecided_amber)} eligible</div>'
                    f'<div style="font-size:0.8rem;color:#64748b;">Medium risk + high confidence, cohort-safe · Reduces curator load ~40%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with c_btn:
                st.markdown('<div class="btn-amber">', unsafe_allow_html=True)
                if st.button(f"Approve ({len(undecided_amber)})", key="as_amber"):
                    st.session_state.pulse_queue_tier = "amber"
                    st.switch_page("pages/1_decision_console.py")
                st.markdown("</div>", unsafe_allow_html=True)
            actions_rendered += 1
        except Exception:
            pass

    # 3. Growth opportunity
    top_pathway = exp_summary.get("top_row")
    if top_pathway is not None:
        try:
            row = top_pathway.to_dict() if hasattr(top_pathway, "to_dict") else top_pathway
            persona_lbl = experiment_persona_label(str(row.get("persona_tier", "Unknown")))
            product_lbl = experiment_product_label(str(row.get("product_code", "Unknown")))
            uplift_pct = _safe_float(row.get("uplift_score"), 0) * 100
            delta_aua = _safe_float(row.get("delta_aua_uplift"), 0)
            c_text, c_btn = st.columns([4, 1])
            with c_text:
                st.markdown(
                    f'<div class="ws-action-card-growth" style="padding:0.6rem 0.9rem;">'
                    f'<div style="font-weight:600;font-size:0.9rem;">Growth — {persona_lbl} + {product_lbl}</div>'
                    f'<div style="font-size:0.8rem;color:#64748b;">+{uplift_pct:.1f}% uplift · +{format_currency(delta_aua)} projected AUA</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with c_btn:
                st.markdown('<div class="btn-growth">', unsafe_allow_html=True)
                if st.button("Inspect Pathway", key="as_growth"):
                    st.switch_page("pages/2_growth_engine.py")
                st.markdown("</div>", unsafe_allow_html=True)
            actions_rendered += 1
        except Exception:
            pass

    if actions_rendered == 0:
        render_empty_state("All Systems Nominal", "No immediate actions required.", "🎯")

    # ── QUEUE PROGRESS — directly under action stack ──
    total_requiring_review = len(undecided_red) + len(undecided_amber)
    total_queue = len(red_cases) + len(amber_cases)
    processed_today = total_queue - total_requiring_review if total_queue else 0
    st.caption(f"Processed: **{processed_today} / {total_queue}** · Remaining: **{total_requiring_review}**")
    st.progress(processed_today / total_queue if total_queue else 1.0)

    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)

    # ── IMPACT — one number + one sentence, CTA right-aligned ──
    st.markdown('<div class="ws-subsection">Impact (rolling 30d)</div>', unsafe_allow_html=True)
    imp_left, imp_right = st.columns([4, 1], vertical_alignment="center")
    with imp_left:
        top_pathway_label = ""
        if top_pathway is not None:
            try:
                row = top_pathway.to_dict() if hasattr(top_pathway, "to_dict") else top_pathway
                top_pathway_label = experiment_product_label(str(row.get("product_code", "—")))
            except Exception:
                pass
        pathway_line = f'<div style="font-size:0.78rem;color:#64748b;">Top pathway: {top_pathway_label}.</div>' if top_pathway_label else ""
        st.markdown(
            f'<div class="ws-impact-card" style="padding:0.6rem 0.9rem;">'
            f'<div style="font-weight:600;font-size:1.0rem;">{format_currency(projected_aua)} projected AUA    ·    Net uplift +{net_uplift:.2f}</div>'
            f'{pathway_line}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with imp_right:
        st.markdown('<div class="btn-muted">', unsafe_allow_html=True)
        if st.button("Growth Engine", key="impact_link"):
            st.switch_page("pages/2_growth_engine.py")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)

    # ── COLLAPSED: System Metrics ──
    with st.expander("System Metrics", expanded=False):
        st.markdown(
            f'<div class="ws-kpi-compact">'
            f'<div class="ws-kpi-compact-item"><span class="val">{format_number(len(filtered))}</span><span class="lbl">Active</span></div>'
            f'<div class="ws-kpi-compact-item"><span class="val">{format_number(total_requiring_review)}</span><span class="lbl">Pending</span></div>'
            f'<div class="ws-kpi-compact-item"><span class="val">{format_number(len(green_cases))}</span><span class="lbl">Auto-Approved</span></div>'
            f'<div class="ws-kpi-compact-item"><span class="val">{format_number(len(red_cases))}</span><span class="lbl">Suppressed</span></div>'
            f'<div class="ws-kpi-compact-item"><span class="val">+{net_uplift:.2f}</span><span class="lbl">Net Uplift</span></div>'
            f'<div class="ws-kpi-compact-item"><span class="val">{format_currency(projected_aua)}</span><span class="lbl">Proj. AUA</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── COLLAPSED: System Status + Model Health ──
    with st.expander("System Status & Governance", expanded=False):
        col_left, col_right = st.columns(2)
        with col_left:
            render_governance_constraints()
            st.markdown(f"""
            <div class="ws-secondary" style="margin-top:0.75rem;">
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
            st.markdown('<div class="btn-muted">', unsafe_allow_html=True)
            if st.button("Compliance Dashboard", key="compliance_link"):
                st.switch_page("pages/3_compliance.py")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("Something went wrong. No data or configuration may be available.")
        st.caption("If this persists, ensure data is generated and paths are correct.")
