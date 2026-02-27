"""
Wealthsimple Pulse -- Financial Curator Dashboard

Human-in-the-Loop traceability UI with:
- Macro Dashboard (BoC Rates, Market Volatility)
- Tiered Queue (Red / Amber / Green governance tabs)
- Batch Approve for Amber cohorts
- Traceable rationale: every recommendation cites a Macro Reason + Behavioral Reason
- Active Learning feedback loop
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.api.feedback import get_feedback_stats, record_feedback, apply_feedback_penalty
from src.api.macro_agent import MacroSnapshot, adjust_confidence_for_macro
from src.classifier.guardrails import apply_guardrails_to_hypothesis
from src.classifier.cohort_engine import build_intent_cohorts
from src.features.nudges import generate_composite_reason, generate_nudge
from src.models.governance import enrich_hypothesis_with_governance
from src.models.predict import XGBSignalModel, predict_signal
from src.utils.io import DATA_EXPERIMENTS, read_parquet, write_parquet

from ui.lib import (
    TIER_LABELS,
    PRODUCT_CODES,
    load_data,
    load_model,
    generate_hypotheses,
    load_cohorts_df,
    save_cohort,
    cohort_metrics,
    get_default_macro,
)

# ---------------------------------------------------------------------------
# Wealthsimple palette tokens
# ---------------------------------------------------------------------------
COIN_WHITE = "#FFFFFF"
MIDNIGHT = "#000000"
WS_GOLD = "#FFB547"
SAGE_GREEN = "#E8F0E8"
STONE_GREY = "#F2F2F2"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wealthsimple Pulse",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)

TIER_COLORS = {
    "aspiring_affluent": "#4ECDC4",
    "sticky_family_leader": "#FFD93D",
    "generation_nerd": "#6C5CE7",
    "not_eligible": "#636E72",
}

SIGNAL_LABELS = {
    "leapfrog_ready": "Leapfrog Signal",
    "liquidity_warning": "Liquidity Watchdog",
    "harvest_opportunity": "Analyst-in-Pocket",
}

GOV_TIER_ICONS = {"green": "🟢", "amber": "🟡", "red": "🔴"}

# ---------------------------------------------------------------------------
# Metric definitions (business + technical) for info icons
# ---------------------------------------------------------------------------
METRIC_DEFINITIONS = {
    "liquid_cash": {
        "label": "Liquid Cash",
        "business": "Estimated immediately accessible cash for the client after accounting for current balances and near-term obligations.",
        "technical": "Approximated as 10% of AUA; would incorporate chequing balance in production.",
    },
    "monthly_burn_rate": {
        "label": "Monthly Burn Rate",
        "business": "Total spending (debits) per month from chequing and credit card.",
        "technical": "Sum of negative amounts over the last 30 days for chequing + credit_card, excluding rent and CC payment transfers.",
    },
    "months_of_runway": {
        "label": "Months of Runway",
        "business": "How many months the client can sustain their recent burn rate before depleting the liquid buffer.",
        "technical": "liquid_cash / spend_velocity_30d.",
    },
    "confidence_score": {
        "label": "Confidence Score",
        "business": "Model probability that this user is exhibiting the detected signal. Higher scores = stronger evidence.",
        "technical": "XGBoost predicted probability, adjusted by macro conditions and active learning feedback.",
    },
    "suggested_amount": {
        "label": "Suggested Amount",
        "business": "Recommended product amount to achieve the stated outcome.",
        "technical": "For Aspiring Affluent: estimated from unused RRSP room and gap to $100k AUA.",
    },
}


def _metric_with_info(metric_key: str, value: str, delta_color: str | None = None):
    """Render a metric with an info popover."""
    defn = METRIC_DEFINITIONS.get(metric_key)
    if defn is None:
        st.metric(metric_key, value)
        return
    col_m, col_i = st.columns([5, 1])
    with col_m:
        if delta_color:
            st.metric(defn["label"], value, delta_color=delta_color)
        else:
            st.metric(defn["label"], value)
    with col_i:
        with st.popover("ℹ"):
            st.markdown("**Business**")
            st.caption(defn["business"])
            st.markdown("**Technical**")
            st.caption(defn["technical"])


# Cohort storage: load_cohorts_df, save_cohort, cohort_metrics from ui.lib


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "decisions" not in st.session_state:
    st.session_state.decisions = {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _inject_ws_theme():
    """Inject global CSS for Wealthsimple-like visual styling."""
    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');

    :root {{
        --ws-midnight: {MIDNIGHT};
        --ws-off-white: {COIN_WHITE};
        --ws-gold: {WS_GOLD};
        --ws-sage: {SAGE_GREEN};
        --ws-stone: {STONE_GREY};
        --ws-radius: 8px;
    }}

    /* App shell */
    .stApp {{
        background-color: var(--ws-off-white);
        color: var(--ws-midnight);
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    section[data-testid="stSidebar"] > div {{
        background-color: var(--ws-stone);
    }}

    /* Typography */
    h1, h2 {{
        font-family: 'Playfair Display', 'Lora', serif;
        letter-spacing: 0.01em;
    }}
    h3, h4, h5, h6, .stMarkdown, .stDataFrame, .stMetric, .stButton > button {{
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Layout helpers */
    .ws-main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 1.5rem 2rem 2.5rem 2rem;
    }}
    .ws-card {{
        background-color: var(--ws-off-white);
        border-radius: var(--ws-radius);
        border: 1px solid var(--ws-stone);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: var(--ws-radius);
        border: 1px solid transparent;
        font-weight: 500;
        padding: 0.35rem 0.9rem;
        transition: all 0.15s ease-out;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }}

    .ws-btn-primary button {{
        background-color: var(--ws-midnight);
        color: var(--ws-off-white);
    }}
    .ws-btn-primary button:hover {{
        background-color: #111111;
    }}

    .ws-btn-danger button {{
        background-color: #FDEDEC;
        color: #C0392B;
        border-color: #F5B7B1;
    }}
    .ws-btn-danger button:hover {{
        background-color: #FADBD8;
    }}

    .ws-btn-secondary button {{
        background-color: var(--ws-stone);
        color: var(--ws-midnight);
    }}
    .ws-btn-secondary button:hover {{
        background-color: #e6e6e6;
    }}

    /* Tables and metrics */
    div[data-testid="stTable"], div[data-testid="stDataFrame"] table {{
        border-radius: var(--ws-radius);
        overflow: hidden;
    }}
    .stMetric > div:first-child {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .stMetric {{
        padding: 0.5rem 0;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def main():
    _inject_ws_theme()
    if "macro" not in st.session_state:
        st.session_state.macro = get_default_macro()

    profiles, txns, features = load_data()
    model = load_model()

    # ---- Sidebar ----
    st.sidebar.title("W Pulse")
    st.sidebar.caption("Financial Curator Dashboard")
    st.sidebar.divider()

    # Decision Summary & Impact (expander)
    with st.sidebar.expander("Decision Summary & Impact", expanded=True):
        approved_decisions = [v for v in st.session_state.decisions.values() if v.get("action") == "approved"]
        if "aum_unlocked_today" not in st.session_state:
            st.session_state.aum_unlocked_today = 0.0
        st.session_state.aum_unlocked_today = len(approved_decisions) * 18000.0  # placeholder
        st.metric("AUM Unlocked (session)", f"${st.session_state.aum_unlocked_today:,.0f}")
        st.metric("Approvals (session)", len(approved_decisions))
        st.progress(min(1.0, len(approved_decisions) / 50.0))
        st.caption("Progress toward daily review goal (50)")
        st.divider()
        n_approved = sum(1 for v in st.session_state.decisions.values() if v["action"] == "approved")
        n_rejected = sum(1 for v in st.session_state.decisions.values() if v["action"] == "rejected")
        n_pending = sum(1 for v in st.session_state.decisions.values() if v["action"] == "pending")
        st.markdown("**Decision Summary**")
        st.metric("Approved", n_approved)
        st.metric("Rejected", n_rejected)
        st.metric("Pending", n_pending)
        fb_stats = get_feedback_stats()
        if fb_stats["total"] > 0:
            st.divider()
            st.markdown("**Active Learning**")
            st.caption(f"Feedback records: {fb_stats['total']}  |  Approval rate: {fb_stats['approval_rate']:.0%}")

    # Filters (expander)
    with st.sidebar.expander("Filters", expanded=True):
        tier_filter = st.multiselect(
            "Filter by Persona Tier",
            options=[k for k in TIER_LABELS if k != "not_eligible"],
            format_func=lambda x: TIER_LABELS[x],
            default=["aspiring_affluent", "sticky_family_leader", "generation_nerd"],
        )
        confidence_min = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)

    # Scenario Planning (expander) — persist macro to session_state
    with st.sidebar.expander("Scenario Planning", expanded=False):
        boc_rate = st.slider("BoC Prime Rate (%)", 3.0, 8.0, float(st.session_state.macro.boc_prime_rate), 0.25, key="sb_boc")
        vix_val = st.slider("Market Volatility (VIX)", 10, 40, int(st.session_state.macro.vix), 1, key="sb_vix")
        st.session_state.macro = MacroSnapshot(boc_prime_rate=boc_rate, vix=vix_val)

    # Cohort Builder (expander) — navigate to page
    with st.sidebar.expander("Cohort Builder", expanded=False):
        if st.button("Open Cohort Builder"):
            st.switch_page("ui/pages/1_cohort_builder.py")

    macro = st.session_state.macro
    hypotheses = generate_hypotheses(features, profiles, model, macro)
    filtered = [
        h for h in hypotheses
        if h["persona_tier"] in tier_filter and h["confidence"] >= confidence_min
    ]
    if macro.boc_prime_rate > 6.0:
        filtered = [h for h in filtered if h["traceability"]["target_product"]["code"] != "RRSP_LOAN"]
        if len(filtered) < len([h for h in hypotheses if h["persona_tier"] in tier_filter and h["confidence"] >= confidence_min]):
            st.sidebar.warning("BoC rate > 6%: Retirement Accelerator suggestions hidden (too risky).")

    # ---- Main content wrapper ----
    st.markdown('<div class="ws-main">', unsafe_allow_html=True)

    # ---- Header ----
    st.title("Wealthsimple Pulse")
    st.caption("AI-Staged Product Recommendations -- Human-in-the-Loop Review")

    # ---- Macro Dashboard ----
    with st.container():
        st.markdown('<div class="ws-card">', unsafe_allow_html=True)
        with st.expander("Macro Dashboard", expanded=True):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("BoC Prime Rate", f"{macro.boc_prime_rate:.2f}%",
                        delta="High" if macro.rates_high else "Normal",
                        delta_color="inverse" if macro.rates_high else "normal")
            mc2.metric("VIX", f"{macro.vix:.1f}",
                        delta="Elevated" if macro.market_volatile else "Normal",
                        delta_color="inverse" if macro.market_volatile else "normal")
            mc3.metric("TSX Volatility", f"{macro.tsx_volatility:.1f}%")
            mc4.metric("Snapshot", macro.timestamp[:10])
        st.markdown("</div>", unsafe_allow_html=True)

    # Summary metrics
    with st.container():
        st.markdown('<div class="ws-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Signals", len(hypotheses))
        col2.metric("Filtered Queue", len(filtered))
        col3.metric("Pending Review", len(filtered) - sum(1 for h in filtered if h["user_id"] in st.session_state.decisions))
        col4.metric("Avg Confidence", f"{sum(h['confidence'] for h in filtered) / max(len(filtered), 1):.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Batch Review by Intent ----
    intent_cohorts = build_intent_cohorts(filtered)
    if intent_cohorts:
        with st.container():
            st.markdown('<div class="ws-card">', unsafe_allow_html=True)
            st.subheader("Batch Review by Intent")
            for cohort in intent_cohorts:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{cohort['name']}** ({len(cohort['user_ids'])} members)")
                    st.caption(cohort["why_summary"])
                    if cohort.get("potential_aum_growth") is not None:
                        st.caption(f"Potential AUM growth: ${cohort['potential_aum_growth']:,.0f}")
                with c2:
                    undecided = [u for u in cohort["user_ids"] if u not in st.session_state.decisions]
                    if undecided and st.button(f"Global Approve ({len(undecided)})", key=f"global_approve_{cohort['intent_id']}"):
                        now = datetime.now(timezone.utc).isoformat()
                        for h in cohort["hypotheses"]:
                            if h["user_id"] in undecided:
                                st.session_state.decisions[h["user_id"]] = {
                                    "action": "approved", "timestamp": now,
                                    "signal": h["signal"], "persona_tier": h["persona_tier"],
                                    "confidence": h["confidence"],
                                }
                                record_feedback(
                                    h["user_id"], h["persona_tier"], h["signal"],
                                    h["traceability"]["target_product"]["code"],
                                    h["confidence"], h.get("governance", {}).get("tier", ""), "approved",
                                    macro_reasons="; ".join(h.get("macro_reasons", [])),
                                )
                        st.success(f"Approved {len(undecided)} recommendations.")
                        st.rerun()
                st.divider()
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("</div>", unsafe_allow_html=True)

    if not filtered:
        st.info("No signals match your filter criteria.")
        return

    # ---- Split into governance tiers ----
    red = [h for h in filtered if h.get("governance", {}).get("tier") == "red"]
    amber = [h for h in filtered if h.get("governance", {}).get("tier") == "amber"]
    green = [h for h in filtered if h.get("governance", {}).get("tier") == "green"]

    # ---- Tiered Queue Tabs ----
    with st.container():
        st.markdown('<div class="ws-card">', unsafe_allow_html=True)
        tab_red, tab_amber, tab_green = st.tabs([
            f"🔴 Red -- Manual Review ({len(red)})",
            f"🟡 Amber -- Batch Review ({len(amber)})",
            f"🟢 Green -- Auto-Approve ({len(green)})",
        ])

        with tab_red:
            _render_queue(red, "red", features)
        with tab_amber:
            _render_amber_queue(amber, features)
        with tab_green:
            _render_queue(green, "green", features)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Queue rendering
# ---------------------------------------------------------------------------
def _build_queue_df(items: list) -> pd.DataFrame:
    rows = []
    for i, h in enumerate(items):
        status = st.session_state.decisions.get(h["user_id"], {}).get("action", "pending")
        gov = h.get("governance", {})
        dist = h.get("distance_to_upgrade") or {}
        path_label = dist.get("cohort_label", "")
        rows.append({
            "idx": i,
            "User ID": h["user_id"][:12] + "...",
            "Tier": TIER_LABELS.get(h["persona_tier"], h["persona_tier"]),
            "Path": path_label,
            "Signal": SIGNAL_LABELS.get(h["signal"], h["signal"]),
            "Confidence": h["confidence"],
            "Gov": f"{GOV_TIER_ICONS.get(gov.get('tier', ''), '')} {gov.get('tier', '').upper()}",
            "Product": h["traceability"]["target_product"]["name"],
            "Status": status.upper(),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _render_queue(items: list, tier_name: str, features: pd.DataFrame):
    """Render a standard queue (Red or Green) with single-item detail."""
    if not items:
        st.info(f"No {tier_name} items.")
        return

    df = _build_queue_df(items)
    selection = st.dataframe(
        df.drop(columns=["idx"]), use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        key=f"queue_{tier_name}",
    )

    selected_rows = selection.selection.rows if selection.selection else []
    if not selected_rows:
        st.caption("Select a row to review.")
        return

    hypothesis = items[selected_rows[0]]
    _render_detail(hypothesis, features)


def _render_amber_queue(items: list, features: pd.DataFrame):
    """Render Amber queue with Batch Approve button."""
    if not items:
        st.info("No amber items.")
        return

    # Batch approve button
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

    df = _build_queue_df(items)
    selection = st.dataframe(
        df.drop(columns=["idx"]), use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        key="queue_amber",
    )

    selected_rows = selection.selection.rows if selection.selection else []
    if not selected_rows:
        st.caption("Select a row for detail, or use Batch Approve above.")
        return

    hypothesis = items[selected_rows[0]]
    _render_detail(hypothesis, features)


# ---------------------------------------------------------------------------
# Detail panel
# ---------------------------------------------------------------------------
def _render_detail(hypothesis: dict, features: pd.DataFrame):
    st.divider()
    st.subheader(f"Traceability Panel -- {hypothesis['user_id'][:16]}...")

    user_id = hypothesis["user_id"]
    gov = hypothesis.get("governance", {})

    # Header row: tier, signal, confidence, governance
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Persona Tier**")
        st.markdown(f":{TIER_COLORS.get(hypothesis['persona_tier'], '#999')}[{TIER_LABELS.get(hypothesis['persona_tier'], '')}]")
    with c2:
        st.markdown("**Detected Signal**")
        st.markdown(f"**{SIGNAL_LABELS.get(hypothesis['signal'], hypothesis['signal'])}**")
    with c3:
        st.markdown("**Confidence Score**")
        _render_confidence_gauge(hypothesis["confidence"])
    with c4:
        st.markdown("**Governance Tier**")
        icon = GOV_TIER_ICONS.get(gov.get("tier", ""), "")
        st.markdown(f"{icon} **{gov.get('label', 'Unknown')}**")
        st.caption(gov.get("reason", ""))
    dist = hypothesis.get("distance_to_upgrade") or {}
    if dist.get("cohort_label"):
        st.caption(f"**Status path:** {dist.get('cohort_label')} — Gap: ${dist.get('gap_dollars', 0):,.0f} to {dist.get('next_milestone_name', '')}")

    # Rationale (Nudge = Behavioral + Macro + Feedback)
    with st.expander("Recommendation Rationale", expanded=True):
        st.markdown(hypothesis.get("nudge", "No rationale available."))

    # Nudge Editor: tone and edit before send
    st.markdown("**Nudge Preview**")
    nudge_tone = st.radio("Tone", ["Professional", "Casual"], horizontal=True, key=f"nudge_tone_{user_id}")
    default_nudge = hypothesis.get("nudge", "")
    edited_nudge = st.text_area("Edit message (optional)", value=default_nudge, height=120, key=f"nudge_edit_{user_id}")
    st.caption("You can edit the message above before approving. Approve & Send records your decision and uses this copy.")

    # Scenario Planning (shared with sidebar / Cohort Builder)
    with st.expander("Scenario Planning", expanded=False):
        macro = st.session_state.get("macro")
        if macro is None:
            from ui.lib import get_default_macro
            st.session_state.macro = get_default_macro()
            macro = st.session_state.macro
        boc_rate = st.slider("BoC Prime Rate (%)", 3.0, 8.0, float(macro.boc_prime_rate), 0.25, key=f"detail_boc_{user_id}")
        vix_val = st.slider("Market Volatility (VIX)", 10, 40, int(macro.vix), 1, key=f"detail_vix_{user_id}")
        st.session_state.macro = MacroSnapshot(boc_prime_rate=boc_rate, vix=vix_val)
        st.caption(f"Current scenario: BoC {boc_rate}%, VIX {vix_val}. Confidence was computed using the active scenario.")

    st.divider()

    # Three traceability panels
    buf_col, prod_col, audit_col = st.columns(3)
    trace = hypothesis["traceability"]

    with buf_col:
        st.markdown("### Spending Buffer")
        sb = trace["spending_buffer"]
        runway = sb["months_of_runway"]
        runway_icon = "normal" if runway >= 6 else ("off" if runway >= 3 else "inverse")
        _metric_with_info("liquid_cash", f"${sb['liquid_cash']:,.0f}")
        _metric_with_info("monthly_burn_rate", f"${sb['monthly_burn_rate']:,.0f}")
        _metric_with_info("months_of_runway", f"{runway:.1f}", delta_color=runway_icon)

    with prod_col:
        st.markdown("### Target Product")
        tp = trace["target_product"]
        st.markdown(f"**{tp['name']}**")
        st.caption(f"Code: `{tp['code']}`")
        if tp.get("projected_yield"):
            st.success(f"Projected Yield: {tp['projected_yield']}")
        if tp.get("suggested_amount"):
            _metric_with_info("suggested_amount", f"${tp['suggested_amount']:,.0f}")

    with audit_col:
        st.markdown("### Audit Log")
        audit = trace["audit_log"]
        audit_df = pd.DataFrame(audit)
        if not audit_df.empty:
            audit_df = audit_df.sort_values("importance", ascending=False)
            audit_df["importance"] = audit_df["importance"].apply(lambda x: f"{x:.4f}")
            audit_df["value"] = audit_df["value"].apply(lambda x: f"{x:,.4f}")
            st.dataframe(audit_df, use_container_width=True, hide_index=True)

    st.divider()

    # Trajectory charts
    user_features = features[features["user_id"] == user_id].sort_values("month")
    if not user_features.empty:
        chart1, chart2 = st.columns(2)
        with chart1:
            fig = go.Figure()
            x_vals = user_features["month"].astype(str)
            y_aua = user_features["aua_current"]
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_aua,
                mode="lines",
                name="AUA",
                line=dict(color=MIDNIGHT, width=1.5),
            ))
            # current value dot
            fig.add_trace(go.Scatter(
                x=[x_vals.iloc[-1]],
                y=[y_aua.iloc[-1]],
                mode="markers",
                marker=dict(color=WS_GOLD, size=8),
                name="Current",
            ))
            fig.add_hline(y=100_000, line_dash="dash", line_color=WS_GOLD, annotation_text="Premium ($100k)")
            fig.add_hline(y=500_000, line_dash="dash", line_color="#999999", annotation_text="Generation ($500k)")
            fig.update_layout(
                title="AUA Over Time",
                xaxis_title="Month",
                yaxis_title="AUA (CAD)",
                height=350,
                showlegend=False,
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig, use_container_width=True)
        with chart2:
            fig2 = go.Figure()
            x_vals2 = user_features["month"].astype(str)
            y_spend = user_features["spend_velocity_30d"]
            fig2.add_trace(go.Scatter(
                x=x_vals2,
                y=y_spend,
                mode="lines",
                name="Spend Velocity",
                line=dict(color=MIDNIGHT, width=1.5),
            ))
            fig2.add_trace(go.Scatter(
                x=[x_vals2.iloc[-1]],
                y=[y_spend.iloc[-1]],
                mode="markers",
                marker=dict(color=WS_GOLD, size=8),
                name="Current",
            ))
            fig2.update_layout(
                title="Spend Velocity (30d)",
                xaxis_title="Month",
                yaxis_title="CAD",
                height=350,
                showlegend=False,
            )
            fig2.update_xaxes(showgrid=False)
            fig2.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ---- Curator Decision ----
    st.divider()
    st.subheader("Curator Decision")

    existing = st.session_state.decisions.get(user_id, {})
    is_locked = existing.get("action") in ("approved", "rejected")

    reject_col, pending_col, approve_col, reason_col = st.columns([1, 1, 1, 2])

    with reject_col:
        st.markdown('<span style="color: #C0392B; font-weight: bold;">Reject</span>', unsafe_allow_html=True)
        st.markdown('<div class="ws-btn-danger">', unsafe_allow_html=True)
        if st.button("Reject", type="secondary", use_container_width=True, disabled=is_locked, key=f"rej_{user_id}"):
            st.session_state.decisions[user_id] = {
                "action": "rejected", "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": hypothesis["signal"], "persona_tier": hypothesis["persona_tier"],
                "confidence": hypothesis["confidence"], "reason": "",
            }
            record_feedback(
                user_id, hypothesis["persona_tier"], hypothesis["signal"],
                hypothesis["traceability"]["target_product"]["code"],
                hypothesis["confidence"], gov.get("tier", ""),
                "rejected", macro_reasons="; ".join(hypothesis.get("macro_reasons", [])),
            )
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with pending_col:
        st.markdown("**Pending / Reconsider**")
        st.markdown('<div class="ws-btn-secondary">', unsafe_allow_html=True)
        if st.button("Mark Pending", use_container_width=True, disabled=is_locked, key=f"pen_{user_id}"):
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

    with approve_col:
        st.markdown(f'<span style="color: {WS_GOLD}; font-weight: bold;">Approve</span>', unsafe_allow_html=True)
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

    with reason_col:
        if existing.get("action") == "rejected":
            reason = st.selectbox(
                "Rejection Reason",
                ["Insufficient evidence", "Client risk concern", "Timing not right", "Macro conditions", "Other"],
                key=f"reason_{user_id}",
            )
            st.session_state.decisions[user_id]["reason"] = reason

    if existing:
        if existing["action"] == "approved":
            st.success(f"APPROVED at {existing['timestamp']}")
        elif existing["action"] == "rejected":
            st.error(f"REJECTED at {existing['timestamp']} -- Reason: {existing.get('reason', 'N/A')}")
        elif existing["action"] == "pending":
            st.info(f"Pending review since {existing.get('timestamp', 'N/A')}. You may still Approve or Reject.")
    if hypothesis.get("guardrail_reasons"):
        st.warning("**Guardrails:** " + " | ".join(hypothesis["guardrail_reasons"]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _confidence_band(confidence: float) -> tuple[str, str]:
    if confidence >= 0.90:
        return "High", "Very strong signal; historically high precision."
    if confidence >= 0.75:
        return "High", "Strong signal; good candidate for approval."
    if confidence >= 0.60:
        return "Medium", "Moderate signal; use additional judgment."
    return "Low", "Weak signal; exploratory only."


def _render_confidence_gauge(confidence: float):
    color = "normal" if confidence >= 0.8 else ("off" if confidence >= 0.6 else "inverse")
    st.metric("Score", f"{confidence:.1%}", delta_color=color)
    band_name, band_desc = _confidence_band(confidence)
    st.caption(f"Band: **{band_name}**")
    st.caption(band_desc)


if __name__ == "__main__":
    main()
