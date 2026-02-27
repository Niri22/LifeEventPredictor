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
import plotly.graph_objects as go
import streamlit as st

from src.api.feedback import get_feedback_stats, record_feedback, apply_feedback_penalty
from src.api.macro_agent import MacroSnapshot, adjust_confidence_for_macro, fetch_macro_snapshot
from src.classifier.persona_classifier import classify_persona_tier
from src.features.nudges import generate_composite_reason, generate_nudge
from src.features.pipeline import build_features
from src.models.governance import enrich_hypothesis_with_governance
from src.models.predict import XGBSignalModel, predict_signal
from src.utils.io import DATA_RAW, DATA_PROCESSED, DATA_EXPERIMENTS, read_parquet, write_parquet

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

TIER_LABELS = {
    "aspiring_affluent": "Aspiring Affluent ($50k-$100k)",
    "sticky_family_leader": "Sticky Family Leader ($100k-$500k)",
    "generation_nerd": "Generation Nerd ($500k+)",
    "not_eligible": "Not Eligible (<$50k)",
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    profiles = read_parquet(DATA_RAW / "user_profiles.parquet")
    txns = read_parquet(DATA_RAW / "transactions.parquet")
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    features = read_parquet(DATA_PROCESSED / "features.parquet")
    features = classify_persona_tier(features)
    return profiles, txns, features


@st.cache_resource
def load_model():
    model = XGBSignalModel()
    model.load_models()
    return model


def generate_hypotheses(
    features: pd.DataFrame,
    profiles: pd.DataFrame,
    model: XGBSignalModel,
    macro: MacroSnapshot,
):
    """Run inference, apply macro + feedback adjustments, governance tiers, and nudges."""
    latest_month = features["month"].max()
    latest = features[features["month"] == latest_month].copy()
    eligible = latest[latest["persona_tier"] != "not_eligible"]

    hypotheses = []
    for _, row in eligible.iterrows():
        feat_dict = row.to_dict()
        persona_tier = row["persona_tier"]
        result = predict_signal(feat_dict, persona_tier, model)
        if result is None:
            continue

        result["user_id"] = row["user_id"]
        profile = profiles[profiles["user_id"] == row["user_id"]]
        if not profile.empty:
            result["age"] = int(profile.iloc[0]["age"])
            result["province"] = profile.iloc[0]["province"]

        product_code = result.get("traceability", {}).get("target_product", {}).get("code", "")
        product_name = result.get("traceability", {}).get("target_product", {}).get("name", "")

        # Macro adjustment
        adj_conf, macro_reasons = adjust_confidence_for_macro(
            result["confidence"], persona_tier, product_code, macro,
        )

        # Feedback penalty
        fb_conf, fb_reason = apply_feedback_penalty(
            adj_conf, persona_tier, result["signal"], product_code,
        )
        result["confidence"] = fb_conf
        result["macro_reasons"] = macro_reasons
        result["feedback_reason"] = fb_reason

        # Nudge
        feat_for_nudge = {**feat_dict}
        if result.get("traceability", {}).get("target_product", {}).get("suggested_amount"):
            feat_for_nudge["suggested_amount"] = result["traceability"]["target_product"]["suggested_amount"]
        behavioral_nudge = generate_nudge(persona_tier, result["signal"], feat_for_nudge, product_name)
        result["nudge"] = generate_composite_reason(behavioral_nudge, macro_reasons, fb_reason)

        # Governance tier
        enrich_hypothesis_with_governance(result)

        hypotheses.append(result)

    return hypotheses


# ---------------------------------------------------------------------------
# Cohort storage (unchanged)
# ---------------------------------------------------------------------------
COHORT_COLUMNS = ["cohort_id", "name", "created_at", "filters"]
MEMBER_COLUMNS = [
    "cohort_id", "user_id", "persona_tier", "signal", "confidence",
    "product_code", "snapshot_month", "age", "province",
]
PRODUCT_CODES = ["RRSP_LOAN", "SUMMIT_PORTFOLIO", "AI_RESEARCH_DIRECT_INDEX"]


def _load_cohorts_df():
    path = DATA_EXPERIMENTS / "cohorts.parquet"
    if not path.exists():
        return pd.DataFrame(columns=COHORT_COLUMNS)
    return read_parquet(path)


def _load_cohort_members_df():
    path = DATA_EXPERIMENTS / "cohort_members.parquet"
    if not path.exists():
        return pd.DataFrame(columns=MEMBER_COLUMNS)
    return read_parquet(path)


def _save_cohort(cohort_id: str, name: str, filters: dict, members: list):
    DATA_EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    cohorts_path = DATA_EXPERIMENTS / "cohorts.parquet"
    members_path = DATA_EXPERIMENTS / "cohort_members.parquet"
    cohorts_df = _load_cohorts_df()
    members_df = _load_cohort_members_df()
    now = datetime.now(timezone.utc).isoformat()
    new_cohort = pd.DataFrame([{
        "cohort_id": cohort_id, "name": name, "created_at": now,
        "filters": json.dumps(filters),
    }])
    cohorts_df = pd.concat([cohorts_df, new_cohort], ignore_index=True)
    members_df = pd.concat([members_df, pd.DataFrame(members)], ignore_index=True)
    write_parquet(cohorts_df, cohorts_path)
    write_parquet(members_df, members_path)


def _cohort_metrics(cohort_id: str, decisions: dict):
    members = _load_cohort_members_df()
    members = members[members["cohort_id"] == cohort_id]
    if members.empty:
        return {"size": 0, "approved": 0, "rejected": 0, "pending": 0, "avg_confidence": 0.0}
    approved = sum(1 for uid in members["user_id"] if decisions.get(uid, {}).get("action") == "approved")
    rejected = sum(1 for uid in members["user_id"] if decisions.get(uid, {}).get("action") == "rejected")
    pending_c = sum(1 for uid in members["user_id"] if decisions.get(uid, {}).get("action") == "pending")
    return {
        "size": len(members), "approved": approved, "rejected": rejected,
        "pending": pending_c, "avg_confidence": round(members["confidence"].mean(), 3),
        "approval_rate": round(approved / max(approved + rejected, 1), 3),
    }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "decisions" not in st.session_state:
    st.session_state.decisions = {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    profiles, txns, features = load_data()
    model = load_model()
    macro = fetch_macro_snapshot()

    # ---- Sidebar ----
    st.sidebar.title("W Pulse")
    st.sidebar.caption("Financial Curator Dashboard")
    st.sidebar.divider()

    tier_filter = st.sidebar.multiselect(
        "Filter by Persona Tier",
        options=[k for k in TIER_LABELS if k != "not_eligible"],
        format_func=lambda x: TIER_LABELS[x],
        default=["aspiring_affluent", "sticky_family_leader", "generation_nerd"],
    )
    confidence_min = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)

    st.sidebar.divider()
    st.sidebar.markdown("**Decision Summary**")
    n_approved = sum(1 for v in st.session_state.decisions.values() if v["action"] == "approved")
    n_rejected = sum(1 for v in st.session_state.decisions.values() if v["action"] == "rejected")
    n_pending = sum(1 for v in st.session_state.decisions.values() if v["action"] == "pending")
    st.sidebar.metric("Approved", n_approved)
    st.sidebar.metric("Rejected", n_rejected)
    st.sidebar.metric("Pending", n_pending)

    # Feedback stats
    fb_stats = get_feedback_stats()
    if fb_stats["total"] > 0:
        st.sidebar.divider()
        st.sidebar.markdown("**Active Learning**")
        st.sidebar.caption(f"Feedback records: {fb_stats['total']}  |  Approval rate: {fb_stats['approval_rate']:.0%}")

    # Cohort builder (sidebar)
    st.sidebar.divider()
    st.sidebar.markdown("**Cohort Builder**")
    cohort_name = st.sidebar.text_input("Cohort name", placeholder="e.g. High-conf Aspiring Affluent")
    cohort_tiers = st.sidebar.multiselect(
        "Persona tiers (cohort)",
        options=[k for k in TIER_LABELS if k != "not_eligible"],
        default=[k for k in TIER_LABELS if k != "not_eligible"],
        format_func=lambda x: TIER_LABELS[x],
        key="cb_tiers",
    )
    conf_low, conf_high = st.sidebar.slider("Confidence range (cohort)", 0.0, 1.0, (0.5, 1.0), 0.05, key="cb_conf")
    cohort_products = st.sidebar.multiselect("Products (cohort)", options=PRODUCT_CODES, default=PRODUCT_CODES, key="cb_prod")

    # Generate hypotheses
    hypotheses = generate_hypotheses(features, profiles, model, macro)
    filtered = [
        h for h in hypotheses
        if h["persona_tier"] in tier_filter and h["confidence"] >= confidence_min
    ]

    if st.sidebar.button("Create Cohort"):
        subset = [
            h for h in filtered
            if h["persona_tier"] in cohort_tiers
            and h["traceability"]["target_product"]["code"] in cohort_products
            and conf_low <= h["confidence"] <= conf_high
        ]
        if not cohort_name.strip():
            st.sidebar.warning("Enter a cohort name.")
        elif not subset:
            st.sidebar.warning("No hypotheses match filters.")
        else:
            cid = str(uuid.uuid4())
            latest_month = features["month"].max() if not features.empty else ""
            _save_cohort(cid, cohort_name.strip(), {
                "persona_tiers": cohort_tiers,
                "confidence_range": [conf_low, conf_high],
                "products": cohort_products,
            }, [
                {
                    "cohort_id": cid, "user_id": h["user_id"], "persona_tier": h["persona_tier"],
                    "signal": h["signal"], "confidence": float(h["confidence"]),
                    "product_code": h["traceability"]["target_product"]["code"],
                    "snapshot_month": str(latest_month), "age": h.get("age"), "province": h.get("province"),
                } for h in subset
            ])
            st.sidebar.success(f"Cohort '{cohort_name.strip()}' created ({len(subset)} members).")

    # ---- Header ----
    st.title("Wealthsimple Pulse")
    st.caption("AI-Staged Product Recommendations -- Human-in-the-Loop Review")

    # ---- Macro Dashboard ----
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

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Signals", len(hypotheses))
    col2.metric("Filtered Queue", len(filtered))
    col3.metric("Pending Review", len(filtered) - sum(1 for h in filtered if h["user_id"] in st.session_state.decisions))
    col4.metric("Avg Confidence", f"{sum(h['confidence'] for h in filtered) / max(len(filtered), 1):.2f}")

    # Cohort explorer
    with st.expander("Cohort Explorer", expanded=False):
        cohorts_df = _load_cohorts_df()
        if cohorts_df.empty:
            st.caption("No cohorts yet.")
        else:
            selected_name = st.selectbox("Select cohort", options=cohorts_df["name"].tolist())
            if selected_name:
                crow = cohorts_df[cohorts_df["name"] == selected_name].iloc[0]
                metrics = _cohort_metrics(crow["cohort_id"], st.session_state.decisions)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Size", metrics["size"])
                m2.metric("Approved", metrics["approved"])
                m3.metric("Rejected", metrics["rejected"])
                m4.metric("Pending", metrics["pending"])

    st.divider()

    if not filtered:
        st.info("No signals match your filter criteria.")
        return

    # ---- Split into governance tiers ----
    red = [h for h in filtered if h.get("governance", {}).get("tier") == "red"]
    amber = [h for h in filtered if h.get("governance", {}).get("tier") == "amber"]
    green = [h for h in filtered if h.get("governance", {}).get("tier") == "green"]

    # ---- Tiered Queue Tabs ----
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


# ---------------------------------------------------------------------------
# Queue rendering
# ---------------------------------------------------------------------------
def _build_queue_df(items: list) -> pd.DataFrame:
    rows = []
    for i, h in enumerate(items):
        status = st.session_state.decisions.get(h["user_id"], {}).get("action", "pending")
        gov = h.get("governance", {})
        rows.append({
            "idx": i,
            "User ID": h["user_id"][:12] + "...",
            "Tier": TIER_LABELS.get(h["persona_tier"], h["persona_tier"]),
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

    # Rationale (Nudge = Behavioral + Macro + Feedback)
    with st.expander("Recommendation Rationale", expanded=True):
        st.markdown(hypothesis.get("nudge", "No rationale available."))

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
            fig.add_trace(go.Scatter(
                x=user_features["month"].astype(str), y=user_features["aua_current"],
                mode="lines+markers", name="AUA", line=dict(color="#6C5CE7", width=2),
            ))
            fig.add_hline(y=100_000, line_dash="dash", line_color="orange", annotation_text="Premium ($100k)")
            fig.add_hline(y=500_000, line_dash="dash", line_color="red", annotation_text="Generation ($500k)")
            fig.update_layout(title="AUA Over Time", xaxis_title="Month", yaxis_title="AUA (CAD)", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with chart2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=user_features["month"].astype(str), y=user_features["spend_velocity_30d"],
                mode="lines+markers", name="Spend Velocity", line=dict(color="#FF6B6B", width=2),
            ))
            fig2.update_layout(title="Spend Velocity (30d)", xaxis_title="Month", yaxis_title="CAD", height=350)
            st.plotly_chart(fig2, use_container_width=True)

    # ---- Curator Decision ----
    st.divider()
    st.subheader("Curator Decision")

    existing = st.session_state.decisions.get(user_id, {})
    is_locked = existing.get("action") in ("approved", "rejected")

    reject_col, pending_col, approve_col, reason_col = st.columns([1, 1, 1, 2])

    with reject_col:
        st.markdown('<span style="color: #e74c3c; font-weight: bold;">Reject</span>', unsafe_allow_html=True)
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

    with pending_col:
        st.markdown("**Pending / Reconsider**")
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

    with approve_col:
        st.markdown('<span style="color: #27ae60; font-weight: bold;">Approve</span>', unsafe_allow_html=True)
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
