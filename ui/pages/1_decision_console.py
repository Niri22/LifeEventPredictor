"""
Decision Console — Case-level review surface.

Tiered Queue (Red / Amber / Green) with a reorganized detail panel:
  Top    — Case Identity (persona, signal, confidence, governance, risk flags)
  Middle — Business Impact (AUA delta, retention, liquidity, macro adjustments)
  Bottom — Decision (rationale, Approve/Reject/Escalate, safety brakes)
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.api.feedback import record_feedback
try:
    from src.api.feedback import get_recent_feedback
except ImportError:
    def get_recent_feedback(limit: int = 10) -> list:
        """Fallback if deployed version lacks this function."""
        return []
from ui.lib import (
    TIER_LABELS,
    SIGNAL_LABELS,
    GOV_TIER_ICONS,
    MIDNIGHT,
    WS_GOLD,
    inject_ws_theme,
    load_data,
    load_precomputed_hypotheses,
    get_default_macro,
    apply_experiment_reweight,
    confidence_band,
    metric_with_info,
    render_confidence_gauge,
    render_pulse_sidebar,
    show_micro_feedback_toast,
    get_system_timestamps,
    render_audit_status,
    render_empty_state,
    get_compliance_info,
    render_client_snapshot,
    format_currency,
)

st.set_page_config(page_title="Decision Console — W Pulse", page_icon="W", layout="wide")

# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------
if "decisions" not in st.session_state:
    st.session_state.decisions = {}
if "macro" not in st.session_state:
    st.session_state.macro = get_default_macro()
if "pulse_queue_tier" not in st.session_state:
    st.session_state.pulse_queue_tier = "red"
if "pulse_confirm_bulk" not in st.session_state:
    st.session_state.pulse_confirm_bulk = False


def _ensure_hypotheses():
    """Use session state if Control Center already ran; else load pre-computed hypotheses and apply filters."""
    if "filtered" in st.session_state and st.session_state["filtered"]:
        return st.session_state["filtered"]
    macro = st.session_state.macro
    hyps = load_precomputed_hypotheses()
    tier_filter = st.session_state.get("pulse_tier_filter", [k for k in TIER_LABELS if k != "not_eligible"])
    confidence_min = st.session_state.get("pulse_confidence_min", 0.5)
    filtered = [h for h in hyps if h["persona_tier"] in tier_filter and h["confidence"] >= confidence_min]
    if macro.boc_prime_rate > 6.0:
        filtered = [h for h in filtered if h.get("traceability", {}).get("target_product", {}).get("code") != "RRSP_LOAN"]
    st.session_state["hypotheses"] = hyps
    st.session_state["filtered"] = filtered
    return filtered


# ---------------------------------------------------------------------------
# Case card — compact operational card (no dataframe)
# ---------------------------------------------------------------------------
def _confidence_label(confidence: float) -> str:
    """Format as 0.00 (Band)."""
    band_name, _ = confidence_band(confidence)
    return f"{confidence:.2f} ({band_name})"


def render_case_card(hypothesis: dict, features: pd.DataFrame, index: int):
    """Render one case as a compact card: left (tier, confidence, persona), center (id, signal, pathway, product, why), right (actions)."""
    user_id = hypothesis["user_id"]
    gov = hypothesis.get("governance", {})
    tier = gov.get("tier", "green")
    trace = hypothesis.get("traceability", {})
    tp = trace.get("target_product", {})
    dist = hypothesis.get("distance_to_upgrade") or {}
    existing = st.session_state.decisions.get(user_id, {})
    is_locked = existing.get("action") in ("approved", "rejected")
    card_key = f"card_{tier}_{user_id}_{index}"

    # Card container — use st.container so Streamlit actually groups the children
    tier_marker = f"cc-tier-{tier}" if tier in ("red", "amber", "green") else "cc-tier-green"
    with st.container(border=True):
        st.markdown(f'<div class="{tier_marker}"></div>', unsafe_allow_html=True)
        col_left, col_center, col_right = st.columns([1.2, 2.5, 1.8])

        with col_left:
            tier_badge = {"red": "🔴 Red", "amber": "🟠 Amber", "green": "🟢 Green"}.get(tier, "Green")
            st.caption(f"**{tier_badge}**")
            conf = hypothesis.get("confidence", 0)
            st.caption(_confidence_label(conf))
            st.caption(TIER_LABELS.get(hypothesis.get("persona_tier", ""), hypothesis.get("persona_tier", "—")))

        with col_center:
            short_id = (user_id[:14] + "…") if len(user_id) > 14 else user_id
            st.caption(f"**{short_id}** · {SIGNAL_LABELS.get(hypothesis.get('signal', ''), hypothesis.get('signal', '—'))}")
            pathway = dist.get("cohort_label") or "—"
            product_name = tp.get("name") or tp.get("code") or "—"
            st.caption(f"Pathway: {pathway} · {product_name}")
            why = hypothesis.get("nudge") or gov.get("reason") or "No rationale."
            if len(why) > 120:
                why = why[:117] + "..."
            st.caption(why)

        with col_right:
            if is_locked:
                action = existing.get("action", "").upper()
                st.success(f"✓ {action}")
            else:
                b_rej, b_app = st.columns(2)
                with b_rej:
                    if st.button("Reject", key=f"rej_{card_key}", use_container_width=True):
                        st.session_state.decisions[user_id] = {
                            "action": "rejected", "timestamp": datetime.now(timezone.utc).isoformat(),
                            "signal": hypothesis["signal"], "persona_tier": hypothesis["persona_tier"],
                            "confidence": hypothesis["confidence"],
                        }
                        record_feedback(
                            user_id, hypothesis["persona_tier"], hypothesis["signal"],
                            tp.get("code", ""), hypothesis["confidence"], gov.get("tier", ""), "rejected",
                            macro_reasons="; ".join(hypothesis.get("macro_reasons", [])),
                        )
                        show_micro_feedback_toast("Rejected")
                        st.rerun()
                with b_app:
                    if st.button("Approve", key=f"app_{card_key}", type="primary", use_container_width=True):
                        st.session_state.decisions[user_id] = {
                            "action": "approved", "timestamp": datetime.now(timezone.utc).isoformat(),
                            "signal": hypothesis["signal"], "persona_tier": hypothesis["persona_tier"],
                            "confidence": hypothesis["confidence"],
                        }
                        record_feedback(
                            user_id, hypothesis["persona_tier"], hypothesis["signal"],
                            tp.get("code", ""), hypothesis["confidence"], gov.get("tier", ""), "approved",
                            macro_reasons="; ".join(hypothesis.get("macro_reasons", [])),
                        )
                        show_micro_feedback_toast("Approved")
                        st.rerun()

        # Inline expand: View Details
        with st.expander("View Details ▾", expanded=False):
            audit = trace.get("audit_log", [])
            if audit:
                top_features = sorted(audit, key=lambda x: x.get("importance", 0), reverse=True)[:5]
                st.caption("**Feature contributions:** " + ", ".join(f"{a.get('feature', '')} ({float(a.get('importance', 0)):.2f})" for a in top_features))
            st.caption(f"**Governance:** {gov.get('label', '')} — {gov.get('reason', '—')}")
            if hypothesis.get("macro_reasons"):
                st.caption("**Macro impact:** " + "; ".join(hypothesis["macro_reasons"][:3]))
            suggested = tp.get("suggested_amount")
            if suggested is not None:
                st.caption(f"**Projected AUA impact:** {format_currency(float(suggested))}")


# ---------------------------------------------------------------------------
# Detail panel — 3 zones: Case Identity / Business Impact / Decision
# ---------------------------------------------------------------------------
def _render_detail(hypothesis: dict, features: pd.DataFrame):
    st.divider()
    user_id = hypothesis["user_id"]
    gov = hypothesis.get("governance", {})
    existing = st.session_state.decisions.get(user_id, {})
    last_reviewed = existing.get("timestamp", "") if existing else None
    if last_reviewed:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(last_reviewed.replace("Z", "+00:00"))
            last_reviewed = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

    # Client Snapshot — full picture in <5 seconds
    render_client_snapshot(hypothesis, features, last_reviewed=last_reviewed)
    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)

    # ======================================================================
    # Case Identity (detailed)
    # ======================================================================
    st.markdown("##### Case Identity")
    ci1, ci2, ci3, ci4 = st.columns(4)
    with ci1:
        st.markdown("**Persona**")
        st.caption(TIER_LABELS.get(hypothesis["persona_tier"], hypothesis.get("persona_tier", "—")))
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
        gap_val = dist.get("gap_dollars", 0) or 0
        st.write("**Status path:**", dist["cohort_label"], "— Gap:", format_currency(gap_val), "to", dist.get("next_milestone_name", ""))

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
            metric_with_info("suggested_amount", format_currency(float(tp["suggested_amount"])))
        audit = trace.get("audit_log", [])
        if audit:
            top_features = sorted(audit, key=lambda x: x.get("importance", 0), reverse=True)[:3]
            st.caption("Top drivers: " + ", ".join(f"{a['feature']} ({float(a['importance']):.3f})" for a in top_features))

    with imp2:
        st.markdown("**Liquidity Snapshot**")
        runway = sb["months_of_runway"]
        runway_icon = "normal" if runway >= 6 else ("off" if runway >= 3 else "inverse")
        metric_with_info("liquid_cash", format_currency(float(sb["liquid_cash"])))
        metric_with_info("monthly_burn_rate", format_currency(float(sb["monthly_burn_rate"])))
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

    btn1, btn2 = st.columns(2)
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

    render_pulse_sidebar("decision")
    tier_filter = st.session_state.get("pulse_tier_filter", [k for k in TIER_LABELS if k != "not_eligible"])
    confidence_min = st.session_state.get("pulse_confidence_min", 0.5)

    # Data
    profiles, _txns, features = load_data()
    filtered = _ensure_hypotheses()
    filtered = [
        h for h in filtered
        if h["persona_tier"] in tier_filter and h["confidence"] >= confidence_min
    ]

    # Header with strong typography
    st.markdown('<div class="ws-main">', unsafe_allow_html=True)
    
    col_title, col_updated = st.columns([3, 1])
    with col_title:
        st.markdown('<h1 class="ws-page-title">Decision Console</h1>', unsafe_allow_html=True)
        st.markdown('<div class="ws-secondary">Case-level review with governance tiers and audit trail</div>', unsafe_allow_html=True)
    with col_updated:
        timestamps = get_system_timestamps()
        compliance = get_compliance_info()
        st.markdown(f"""
        <div class="ws-micro" style="text-align: right;">
        <div>Last updated: {timestamps['last_updated']}</div>
        <div>Model: {compliance['model_version']} | Decisions: 100% logged</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="ws-divider"></div>', unsafe_allow_html=True)

    if not filtered:
        render_empty_state(
            "No Cases Match Filters",
            "Adjust your persona tier or confidence thresholds to see cases.",
            "🔍"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Executive-grade Audit Status panel (compact, above the fold)
    render_audit_status()

    red = [h for h in filtered if h.get("governance", {}).get("tier") == "red"]
    amber = [h for h in filtered if h.get("governance", {}).get("tier") == "amber"]
    green = [h for h in filtered if h.get("governance", {}).get("tier") == "green"]
    tier_counts = {"red": len(red), "amber": len(amber), "green": len(green)}
    current_tier = st.session_state.pulse_queue_tier
    items_by_tier = {"red": red, "amber": amber, "green": green}
    queue_items = items_by_tier.get(current_tier, [])

    # Tier segmented control (no full page reload — session state)
    st.markdown('<div class="ws-section-header">Case Queue</div>', unsafe_allow_html=True)
    seg_r, seg_a, seg_g = st.columns(3)
    with seg_r:
        if st.button(f"🔴 Red ({tier_counts['red']})", key="seg_red", use_container_width=True, type="primary" if current_tier == "red" else "secondary"):
            st.session_state.pulse_queue_tier = "red"
            st.rerun()
    with seg_a:
        if st.button(f"🟠 Amber ({tier_counts['amber']})", key="seg_amber", use_container_width=True, type="primary" if current_tier == "amber" else "secondary"):
            st.session_state.pulse_queue_tier = "amber"
            st.rerun()
    with seg_g:
        if st.button(f"🟢 Green ({tier_counts['green']})", key="seg_green", use_container_width=True, type="primary" if current_tier == "green" else "secondary"):
            st.session_state.pulse_queue_tier = "green"
            st.rerun()

    # Review progress: require review = undecided in this tier
    decided = st.session_state.decisions
    in_tier = queue_items
    undecided_in_tier = [h for h in in_tier if h["user_id"] not in decided]
    processed_in_tier = len(in_tier) - len(undecided_in_tier)
    total_in_tier = len(in_tier)
    st.markdown(f'<div class="queue-progress"><strong>{len(undecided_in_tier)}</strong> cases require review · Processed: <strong>{processed_in_tier} / {total_in_tier}</strong></div>', unsafe_allow_html=True)
    if total_in_tier > 0:
        st.progress(processed_in_tier / total_in_tier)

    # Bulk Approve All Eligible (Amber only), with confirmation
    if current_tier == "amber" and undecided_in_tier:
        if not st.session_state.pulse_confirm_bulk:
            if st.button(f"Approve All Eligible ({len(undecided_in_tier)} cases)", type="primary", key="bulk_approve_btn"):
                st.session_state.pulse_confirm_bulk = True
                st.rerun()
        else:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Confirm — Approve All", type="primary", key="bulk_confirm_yes"):
                    now = datetime.now(timezone.utc).isoformat()
                    for h in undecided_in_tier:
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
                    st.session_state.pulse_confirm_bulk = False
                    show_micro_feedback_toast(f"Approved {len(undecided_in_tier)} cases")
                    st.rerun()
            with c2:
                if st.button("Cancel", key="bulk_confirm_no"):
                    st.session_state.pulse_confirm_bulk = False
                    st.rerun()
            st.caption("Confirm bulk approval for all eligible Amber cases above.")

    # Case cards — paginated to avoid widget overload
    PAGE_SIZE = 20
    if not queue_items:
        st.info(f"No {current_tier} cases in queue.")
    else:
        page_key = f"page_{current_tier}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 0
        total_pages = max(1, (len(queue_items) + PAGE_SIZE - 1) // PAGE_SIZE)
        current_page = st.session_state[page_key]
        start = current_page * PAGE_SIZE
        end = min(start + PAGE_SIZE, len(queue_items))
        page_items = queue_items[start:end]

        for i, h in enumerate(page_items):
            try:
                render_case_card(h, features, start + i)
            except Exception:
                st.caption(f"Could not render case {h.get('user_id', '?')}")

        if total_pages > 1:
            pg_prev, pg_info, pg_next = st.columns([1, 2, 1])
            with pg_info:
                st.caption(f"Page {current_page + 1} of {total_pages} · Showing {start + 1}–{end} of {len(queue_items)}")
            with pg_prev:
                if current_page > 0:
                    if st.button("← Previous", key=f"prev_{current_tier}"):
                        st.session_state[page_key] = current_page - 1
                        st.rerun()
            with pg_next:
                if current_page < total_pages - 1:
                    if st.button("Next →", key=f"next_{current_tier}"):
                        st.session_state[page_key] = current_page + 1
                        st.rerun()

    # Audit Log — last 10 actions (from SQLite or session)
    st.markdown('<div class="ws-section-header" style="margin-top: 1.5rem;">Audit Log</div>', unsafe_allow_html=True)
    try:
        recent = get_recent_feedback(limit=10)
    except Exception:
        recent = []
    if not recent:
        decisions = st.session_state.get("decisions", {})
        items = [
            {"timestamp": v.get("timestamp", ""), "user_id": k, "action": v.get("action", ""), "governance_tier": v.get("persona_tier", ""), "reason": ""}
            for k, v in list(decisions.items())[-10:]
        ]
        items.reverse()
    else:
        items = recent
    if items:
        log_df = pd.DataFrame(items)
        log_df = log_df.rename(columns={"timestamp": "Timestamp", "user_id": "User ID", "action": "Action", "governance_tier": "Tier", "reason": "Reason"})
        st.dataframe(log_df, use_container_width=True, hide_index=True, height=280)
    else:
        st.caption("No audit entries yet. Decisions will appear here.")

    st.markdown("</div>", unsafe_allow_html=True)


try:
    main()
except Exception:
    st.error("Something went wrong. Check filters or data availability.")
    st.info("If this persists, check logs.")
