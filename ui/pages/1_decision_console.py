"""
Decision Console — Case-level review surface.

Tiered Queue (Red / Amber / Green) with a reorganized detail panel:
  Top    — Case Identity (persona, signal, confidence, governance, risk flags)
  Middle — Business Impact (AUA delta, retention, liquidity, macro adjustments)
  Bottom — Decision (rationale, Approve/Reject/Escalate, safety brakes)
"""

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.api.feedback import record_feedback
from src.api.macro_agent import MacroSnapshot

from ui.lib import (
    TIER_LABELS,
    TIER_COLORS,
    SIGNAL_LABELS,
    GOV_TIER_ICONS,
    MIDNIGHT,
    WS_GOLD,
    inject_ws_theme,
    load_data,
    load_model,
    generate_hypotheses,
    get_default_macro,
    apply_experiment_reweight,
    build_queue_df,
    metric_with_info,
    render_confidence_gauge,
    confidence_band,
)

st.set_page_config(page_title="Decision Console — W Pulse", page_icon="W", layout="wide")

# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------
if "decisions" not in st.session_state:
    st.session_state.decisions = {}
if "macro" not in st.session_state:
    st.session_state.macro = get_default_macro()


def _ensure_hypotheses():
    """Regenerate hypotheses if the Control Center hasn't populated them yet."""
    if "filtered" in st.session_state and st.session_state["filtered"]:
        return st.session_state["filtered"]
    profiles, _txns, features = load_data()
    model = load_model()
    macro = st.session_state.macro
    hyps = generate_hypotheses(features, profiles, model, macro)
    st.session_state["hypotheses"] = hyps
    st.session_state["filtered"] = hyps
    return hyps


# ---------------------------------------------------------------------------
# Queue renderers
# ---------------------------------------------------------------------------
def _render_queue(items: list, tier_name: str, features: pd.DataFrame):
    if not items:
        st.info(f"No {tier_name} items.")
        return
    df = build_queue_df(items)
    selection = st.dataframe(
        df.drop(columns=["idx"]), use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        key=f"queue_{tier_name}",
    )
    selected_rows = selection.selection.rows if selection.selection else []
    if not selected_rows:
        st.caption("Select a row to review.")
        return
    _render_detail(items[selected_rows[0]], features)


def _render_amber_queue(items: list, features: pd.DataFrame):
    if not items:
        st.info("No amber items.")
        return
    undecided = [h for h in items if h["user_id"] not in st.session_state.decisions]
    if undecided:
        if st.button(f"Batch Approve All {len(undecided)} Amber Items", type="primary", key="batch_approve"):
            now = datetime.now(timezone.utc).isoformat()
            for h in undecided:
                st.session_state.decisions[h["user_id"]] = {
                    "action": "approved", "timestamp": now,
                    "signal": h["signal"], "persona_tier": h["persona_tier"],
                    "confidence": h["confidence"],
                }
                record_feedback(
                    h["user_id"], h["persona_tier"], h["signal"],
                    h["traceability"]["target_product"]["code"],
                    h["confidence"], "amber", "approved",
                    macro_reasons="; ".join(h.get("macro_reasons", [])),
                )
            st.success(f"Batch approved {len(undecided)} amber recommendations.")
            st.rerun()
    df = build_queue_df(items)
    selection = st.dataframe(
        df.drop(columns=["idx"]), use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        key="queue_amber",
    )
    selected_rows = selection.selection.rows if selection.selection else []
    if not selected_rows:
        st.caption("Select a row for detail, or use Batch Approve above.")
        return
    _render_detail(items[selected_rows[0]], features)


# ---------------------------------------------------------------------------
# Detail panel — 3 zones: Case Identity / Business Impact / Decision
# ---------------------------------------------------------------------------
def _render_detail(hypothesis: dict, features: pd.DataFrame):
    st.divider()
    user_id = hypothesis["user_id"]
    gov = hypothesis.get("governance", {})

    # ======================================================================
    # TOP — Case Identity
    # ======================================================================
    st.markdown("##### Case Identity")
    ci1, ci2, ci3, ci4 = st.columns(4)
    with ci1:
        st.markdown("**Persona**")
        st.markdown(f":{TIER_COLORS.get(hypothesis['persona_tier'], '#999')}[{TIER_LABELS.get(hypothesis['persona_tier'], '')}]")
    with ci2:
        st.markdown("**Signal**")
        st.markdown(f"**{SIGNAL_LABELS.get(hypothesis['signal'], hypothesis['signal'])}**")
    with ci3:
        st.markdown("**Confidence**")
        render_confidence_gauge(hypothesis["confidence"])
    with ci4:
        st.markdown("**Governance Tier**")
        icon = GOV_TIER_ICONS.get(gov.get("tier", ""), "")
        st.markdown(f"{icon} **{gov.get('label', 'Unknown')}**")
        st.caption(gov.get("reason", ""))

    dist = hypothesis.get("distance_to_upgrade") or {}
    if dist.get("cohort_label"):
        st.caption(f"**Status path:** {dist['cohort_label']} — Gap: ${dist.get('gap_dollars', 0):,.0f} to {dist.get('next_milestone_name', '')}")

    if hypothesis.get("guardrail_reasons"):
        st.warning("**Risk Flags:** " + " | ".join(hypothesis["guardrail_reasons"]))

    priority_score, safety_actions = apply_experiment_reweight(hypothesis)
    with st.expander("Priority Score Breakdown", expanded=False):
        st.metric("Calibrated confidence", f"{hypothesis.get('confidence', 0):.2%}")
        if hypothesis.get("macro_reasons"):
            st.caption("Macro modifier: " + "; ".join(hypothesis["macro_reasons"][:2]))
        if hypothesis.get("feedback_reason"):
            st.caption("Feedback penalty: " + (hypothesis["feedback_reason"] or ""))
        st.metric("Priority score (with uplift)", f"{priority_score:.2%}")
        st.caption("Priority = calibrated_confidence × (1 + uplift_weight)")

    st.divider()

    # ======================================================================
    # MIDDLE — Business Impact
    # ======================================================================
    st.markdown("##### Business Impact")
    trace = hypothesis["traceability"]
    sb = trace["spending_buffer"]

    imp1, imp2, imp3 = st.columns(3)
    with imp1:
        st.markdown("**Impact Summary**")
        tp = trace["target_product"]
        st.markdown(f"Product: **{tp['name']}** (`{tp['code']}`)")
        if tp.get("projected_yield"):
            st.success(f"Projected Yield: {tp['projected_yield']}")
        if tp.get("suggested_amount"):
            metric_with_info("suggested_amount", f"${tp['suggested_amount']:,.0f}")
        audit = trace.get("audit_log", [])
        if audit:
            top_features = sorted(audit, key=lambda x: x.get("importance", 0), reverse=True)[:3]
            st.caption("Top drivers: " + ", ".join(f"{a['feature']} ({float(a['importance']):.3f})" for a in top_features))

    with imp2:
        st.markdown("**Liquidity Snapshot**")
        runway = sb["months_of_runway"]
        runway_icon = "normal" if runway >= 6 else ("off" if runway >= 3 else "inverse")
        metric_with_info("liquid_cash", f"${sb['liquid_cash']:,.0f}")
        metric_with_info("monthly_burn_rate", f"${sb['monthly_burn_rate']:,.0f}")
        metric_with_info("months_of_runway", f"{runway:.1f}", delta_color=runway_icon)

    with imp3:
        st.markdown("**Macro Adjustments**")
        if hypothesis.get("macro_reasons"):
            for mr in hypothesis["macro_reasons"]:
                st.caption(f"• {mr}")
        else:
            st.caption("No macro adjustments applied.")
        macro = st.session_state.get("macro")
        if macro:
            st.caption(f"BoC: {macro.boc_prime_rate:.2f}% | VIX: {macro.vix:.0f}")

    # Trajectory charts — collapsed
    user_features = features[features["user_id"] == user_id].sort_values("month")
    if not user_features.empty:
        with st.expander("Trajectory Charts", expanded=False):
            ch1, ch2 = st.columns(2)
            with ch1:
                x_vals = user_features["month"].astype(str)
                y_aua = user_features["aua_current"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_aua, mode="lines", line=dict(color=MIDNIGHT, width=1.5)))
                fig.add_trace(go.Scatter(x=[x_vals.iloc[-1]], y=[y_aua.iloc[-1]], mode="markers", marker=dict(color=WS_GOLD, size=8)))
                fig.add_hline(y=100_000, line_dash="dash", line_color=WS_GOLD, annotation_text="Premium")
                fig.add_hline(y=500_000, line_dash="dash", line_color="#999", annotation_text="Generation")
                fig.update_layout(title="AUA Over Time", height=300, showlegend=False)
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False, zeroline=False)
                st.plotly_chart(fig, use_container_width=True)
            with ch2:
                y_spend = user_features["spend_velocity_30d"]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x_vals, y=y_spend, mode="lines", line=dict(color=MIDNIGHT, width=1.5)))
                fig2.add_trace(go.Scatter(x=[x_vals.iloc[-1]], y=[y_spend.iloc[-1]], mode="markers", marker=dict(color=WS_GOLD, size=8)))
                fig2.update_layout(title="Spend Velocity (30d)", height=300, showlegend=False)
                fig2.update_xaxes(showgrid=False)
                fig2.update_yaxes(showgrid=False, zeroline=False)
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ======================================================================
    # BOTTOM — Decision (visually dominant)
    # ======================================================================
    st.markdown("##### Decision")

    # Safety brake banner
    if safety_actions:
        for sa in safety_actions:
            st.error(f"🛑 **Safety brake:** {sa.get('reason', sa.get('type', 'Unknown'))} — metric: {sa.get('metric', '')}")

    # Recommendation Rationale — full-width, prominent
    st.markdown(
        f'<div style="background:#f8f8f8; border-left:4px solid {WS_GOLD}; padding:0.75rem 1rem; border-radius:4px; margin-bottom:1rem;">'
        f'{hypothesis.get("nudge", "No rationale available.")}</div>',
        unsafe_allow_html=True,
    )

    # Approve / Reject / Escalate
    existing = st.session_state.decisions.get(user_id, {})
    is_locked = existing.get("action") in ("approved", "rejected")

    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        st.markdown('<div class="ws-btn-danger">', unsafe_allow_html=True)
        if st.button("Reject", use_container_width=True, disabled=is_locked, key=f"rej_{user_id}"):
            st.session_state.decisions[user_id] = {
                "action": "rejected", "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": hypothesis["signal"], "persona_tier": hypothesis["persona_tier"],
                "confidence": hypothesis["confidence"],
            }
            record_feedback(
                user_id, hypothesis["persona_tier"], hypothesis["signal"],
                hypothesis["traceability"]["target_product"]["code"],
                hypothesis["confidence"], gov.get("tier", ""), "rejected",
                macro_reasons="; ".join(hypothesis.get("macro_reasons", [])),
            )
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with btn2:
        st.markdown('<div class="ws-btn-secondary">', unsafe_allow_html=True)
        if st.button("Escalate / Pending", use_container_width=True, disabled=is_locked, key=f"esc_{user_id}"):
            st.session_state.decisions[user_id] = {
                "action": "pending", "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": hypothesis["signal"], "persona_tier": hypothesis["persona_tier"],
                "confidence": hypothesis["confidence"],
            }
            record_feedback(
                user_id, hypothesis["persona_tier"], hypothesis["signal"],
                hypothesis["traceability"]["target_product"]["code"],
                hypothesis["confidence"], gov.get("tier", ""), "pending",
            )
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with btn3:
        st.markdown('<div class="ws-btn-primary">', unsafe_allow_html=True)
        if st.button("Approve", type="primary", use_container_width=True, disabled=is_locked, key=f"app_{user_id}"):
            st.session_state.decisions[user_id] = {
                "action": "approved", "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": hypothesis["signal"], "persona_tier": hypothesis["persona_tier"],
                "confidence": hypothesis["confidence"],
            }
            record_feedback(
                user_id, hypothesis["persona_tier"], hypothesis["signal"],
                hypothesis["traceability"]["target_product"]["code"],
                hypothesis["confidence"], gov.get("tier", ""), "approved",
            )
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if existing:
        action = existing["action"]
        ts = existing.get("timestamp", "")
        if action == "approved":
            st.success(f"APPROVED at {ts}")
        elif action == "rejected":
            st.error(f"REJECTED at {ts}")
        elif action == "pending":
            st.info(f"Pending / Escalated since {ts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    inject_ws_theme()

    # Sidebar: mirrors Control Center sidebar
    st.sidebar.title("W Pulse")
    st.sidebar.caption("Decision Console")
    st.sidebar.divider()

    with st.sidebar.expander("Filters", expanded=True):
        tier_filter = st.multiselect(
            "Persona Tier",
            options=[k for k in TIER_LABELS if k != "not_eligible"],
            format_func=lambda x: TIER_LABELS[x],
            default=["aspiring_affluent", "sticky_family_leader", "generation_nerd"],
            key="dc_tier",
        )
        confidence_min = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05, key="dc_conf")

    with st.sidebar.expander("Scenario Planning", expanded=False):
        boc_rate = st.slider("BoC Prime Rate (%)", 3.0, 8.0, float(st.session_state.macro.boc_prime_rate), 0.25, key="dc_boc")
        vix_val = st.slider("Market Volatility (VIX)", 10, 40, int(st.session_state.macro.vix), 1, key="dc_vix")
        st.session_state.macro = MacroSnapshot(boc_prime_rate=boc_rate, vix=vix_val)

    if st.sidebar.button("← Control Center"):
        st.switch_page("app.py")

    # Data
    profiles, _txns, features = load_data()
    filtered = _ensure_hypotheses()
    filtered = [
        h for h in filtered
        if h["persona_tier"] in tier_filter and h["confidence"] >= confidence_min
    ]

    # Header
    st.markdown('<div class="ws-main">', unsafe_allow_html=True)
    st.title("Decision Console")
    st.caption("Tiered Queue — Case-level Review")

    if not filtered:
        st.info("No signals match your filter criteria.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    red = [h for h in filtered if h.get("governance", {}).get("tier") == "red"]
    amber = [h for h in filtered if h.get("governance", {}).get("tier") == "amber"]
    green = [h for h in filtered if h.get("governance", {}).get("tier") == "green"]

    tab_red, tab_amber, tab_green = st.tabs([
        f"🔴 Red — Manual Review ({len(red)})",
        f"🟡 Amber — Batch Review ({len(amber)})",
        f"🟢 Green — Auto-Approve ({len(green)})",
    ])
    with tab_red:
        _render_queue(red, "red", features)
    with tab_amber:
        _render_amber_queue(amber, features)
    with tab_green:
        _render_queue(green, "green", features)

    st.markdown("</div>", unsafe_allow_html=True)


main()
