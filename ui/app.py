"""
Wealthsimple Pulse -- Financial Curator Dashboard

Human-in-the-Loop traceability UI for reviewing AI-staged product recommendations.
Displays: Queue View, Traceability Panel (Spending Buffer, Target Product Yield,
Audit Log), and Approve/Reject flow.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.classifier.persona_classifier import classify_persona_tier
from src.features.pipeline import build_features
from src.models.predict import XGBSignalModel, predict_signal
from src.utils.io import DATA_RAW, DATA_PROCESSED, read_parquet

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

# ---------------------------------------------------------------------------
# Data loading (cached)
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


def generate_hypotheses(features: pd.DataFrame, profiles: pd.DataFrame, model: XGBSignalModel):
    """Run inference on latest month for all eligible users."""
    latest_month = features["month"].max()
    latest = features[features["month"] == latest_month].copy()
    eligible = latest[latest["persona_tier"] != "not_eligible"]

    hypotheses = []
    for _, row in eligible.iterrows():
        feat_dict = row.to_dict()
        persona_tier = row["persona_tier"]
        result = predict_signal(feat_dict, persona_tier, model)
        if result is not None:
            result["user_id"] = row["user_id"]
            profile = profiles[profiles["user_id"] == row["user_id"]]
            if not profile.empty:
                result["age"] = int(profile.iloc[0]["age"])
                result["province"] = profile.iloc[0]["province"]
            hypotheses.append(result)

    return hypotheses


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "decisions" not in st.session_state:
    st.session_state.decisions = {}

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    profiles, txns, features = load_data()
    model = load_model()

    # Sidebar
    st.sidebar.title("W Pulse")
    st.sidebar.caption("Financial Curator Dashboard")
    st.sidebar.divider()

    # Filter controls
    tier_filter = st.sidebar.multiselect(
        "Filter by Persona Tier",
        options=list(TIER_LABELS.keys()),
        format_func=lambda x: TIER_LABELS[x],
        default=["aspiring_affluent", "sticky_family_leader", "generation_nerd"],
    )

    confidence_min = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)

    st.sidebar.divider()
    st.sidebar.markdown("**Decision Summary**")
    approved = sum(1 for v in st.session_state.decisions.values() if v["action"] == "approved")
    rejected = sum(1 for v in st.session_state.decisions.values() if v["action"] == "rejected")
    st.sidebar.metric("Approved", approved)
    st.sidebar.metric("Rejected", rejected)

    # Generate hypotheses
    hypotheses = generate_hypotheses(features, profiles, model)

    # Apply filters
    filtered = [
        h for h in hypotheses
        if h["persona_tier"] in tier_filter and h["confidence"] >= confidence_min
    ]

    # ---------------------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------------------
    st.title("Wealthsimple Pulse")
    st.caption("AI-Staged Product Recommendations -- Human-in-the-Loop Review")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Signals", len(hypotheses))
    col2.metric("Filtered Queue", len(filtered))
    col3.metric("Pending Review", len(filtered) - len([
        h for h in filtered if h["user_id"] in st.session_state.decisions
    ]))
    col4.metric("Avg Confidence", f"{sum(h['confidence'] for h in filtered) / max(len(filtered), 1):.2f}")

    st.divider()

    if not filtered:
        st.info("No signals match your filter criteria.")
        return

    # ---------------------------------------------------------------------------
    # Queue View
    # ---------------------------------------------------------------------------
    queue_data = []
    for i, h in enumerate(filtered):
        status = st.session_state.decisions.get(h["user_id"], {}).get("action", "pending")
        queue_data.append({
            "idx": i,
            "User ID": h["user_id"][:12] + "...",
            "Tier": TIER_LABELS.get(h["persona_tier"], h["persona_tier"]),
            "Signal": SIGNAL_LABELS.get(h["signal"], h["signal"]),
            "Confidence": h["confidence"],
            "Product": h["traceability"]["target_product"]["name"],
            "Status": status.upper(),
        })

    queue_df = pd.DataFrame(queue_data)

    st.subheader("Signal Queue")

    # Clickable table
    selection = st.dataframe(
        queue_df.drop(columns=["idx"]),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = selection.selection.rows if selection.selection else []
    if not selected_rows:
        st.info("Select a row above to view the full traceability panel.")
        return

    selected_idx = selected_rows[0]
    hypothesis = filtered[selected_idx]

    st.divider()

    # ---------------------------------------------------------------------------
    # Detail View -- Traceability Panel
    # ---------------------------------------------------------------------------
    st.subheader(f"Traceability Panel -- {hypothesis['user_id'][:16]}...")

    tier_col, signal_col, conf_col = st.columns(3)
    with tier_col:
        st.markdown(f"**Persona Tier**")
        st.markdown(f":{TIER_COLORS.get(hypothesis['persona_tier'], '#999')}[{TIER_LABELS.get(hypothesis['persona_tier'], '')}]")
    with signal_col:
        st.markdown(f"**Detected Signal**")
        st.markdown(f"**{SIGNAL_LABELS.get(hypothesis['signal'], hypothesis['signal'])}**")
    with conf_col:
        st.markdown(f"**Confidence Score**")
        _render_confidence_gauge(hypothesis["confidence"])

    st.divider()

    # Three traceability panels side by side
    buf_col, prod_col, audit_col = st.columns(3)

    trace = hypothesis["traceability"]

    # --- Spending Buffer ---
    with buf_col:
        st.markdown("### Spending Buffer")
        sb = trace["spending_buffer"]
        runway = sb["months_of_runway"]
        if runway >= 6:
            runway_color = "green"
            runway_icon = "normal"
        elif runway >= 3:
            runway_color = "orange"
            runway_icon = "off"
        else:
            runway_color = "red"
            runway_icon = "inverse"

        st.metric("Liquid Cash", f"${sb['liquid_cash']:,.0f}")
        st.metric("Monthly Burn Rate", f"${sb['monthly_burn_rate']:,.0f}")
        st.metric("Months of Runway", f"{runway:.1f}", delta_color=runway_icon)

    # --- Target Product Yield ---
    with prod_col:
        st.markdown("### Target Product")
        tp = trace["target_product"]
        st.markdown(f"**{tp['name']}**")
        st.caption(f"Code: `{tp['code']}`")
        if tp.get("projected_yield"):
            st.success(f"Projected Yield: {tp['projected_yield']}")
        if tp.get("suggested_amount"):
            st.metric("Suggested Amount", f"${tp['suggested_amount']:,.0f}")

    # --- Audit Log ---
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

    # ---------------------------------------------------------------------------
    # Trajectory Charts
    # ---------------------------------------------------------------------------
    st.subheader("Spending Trajectory")
    user_id = hypothesis["user_id"]
    user_features = features[features["user_id"] == user_id].sort_values("month")

    if not user_features.empty:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_aua = go.Figure()
            fig_aua.add_trace(go.Scatter(
                x=user_features["month"].astype(str),
                y=user_features["aua_current"],
                mode="lines+markers",
                name="AUA",
                line=dict(color="#6C5CE7", width=2),
            ))
            # Threshold lines
            fig_aua.add_hline(y=100_000, line_dash="dash", line_color="orange",
                              annotation_text="Premium ($100k)")
            fig_aua.add_hline(y=500_000, line_dash="dash", line_color="red",
                              annotation_text="Generation ($500k)")
            fig_aua.update_layout(
                title="AUA Over Time",
                xaxis_title="Month",
                yaxis_title="AUA (CAD)",
                height=350,
            )
            st.plotly_chart(fig_aua, use_container_width=True)

        with chart_col2:
            fig_spend = go.Figure()
            fig_spend.add_trace(go.Scatter(
                x=user_features["month"].astype(str),
                y=user_features["spend_velocity_30d"],
                mode="lines+markers",
                name="Spend Velocity",
                line=dict(color="#FF6B6B", width=2),
            ))
            fig_spend.add_trace(go.Scatter(
                x=user_features["month"].astype(str),
                y=user_features["savings_rate"] * user_features.get("monthly_income", user_features["spend_velocity_30d"]),
                mode="lines+markers",
                name="Savings Rate",
                line=dict(color="#4ECDC4", width=2),
                yaxis="y2",
            ))
            fig_spend.update_layout(
                title="Spend Velocity & Savings Rate",
                xaxis_title="Month",
                yaxis_title="Spend (CAD)",
                yaxis2=dict(title="Savings Rate", overlaying="y", side="right"),
                height=350,
            )
            st.plotly_chart(fig_spend, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Action Bar -- Approve / Reject
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("Curator Decision")

    existing = st.session_state.decisions.get(user_id, {})

    action_col1, action_col2, reason_col = st.columns([1, 1, 2])

    with action_col1:
        if st.button(
            "Approve",
            type="primary",
            use_container_width=True,
            disabled=existing.get("action") == "approved",
        ):
            st.session_state.decisions[user_id] = {
                "action": "approved",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": hypothesis["signal"],
                "persona_tier": hypothesis["persona_tier"],
                "confidence": hypothesis["confidence"],
            }
            st.rerun()

    with action_col2:
        if st.button(
            "Reject",
            type="secondary",
            use_container_width=True,
            disabled=existing.get("action") == "rejected",
        ):
            st.session_state.decisions[user_id] = {
                "action": "rejected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": hypothesis["signal"],
                "persona_tier": hypothesis["persona_tier"],
                "confidence": hypothesis["confidence"],
                "reason": "",
            }
            st.rerun()

    with reason_col:
        if existing.get("action") == "rejected":
            reason = st.selectbox(
                "Rejection Reason",
                ["Insufficient evidence", "Client risk concern", "Timing not right", "Other"],
                key=f"reason_{user_id}",
            )
            st.session_state.decisions[user_id]["reason"] = reason

    if existing:
        if existing["action"] == "approved":
            st.success(f"APPROVED at {existing['timestamp']}")
        elif existing["action"] == "rejected":
            st.error(f"REJECTED at {existing['timestamp']} -- Reason: {existing.get('reason', 'N/A')}")


def _render_confidence_gauge(confidence: float):
    """Render a confidence score as a colored metric."""
    if confidence >= 0.8:
        color = "normal"
    elif confidence >= 0.6:
        color = "off"
    else:
        color = "inverse"
    st.metric("Score", f"{confidence:.1%}", delta_color=color)


if __name__ == "__main__":
    main()
