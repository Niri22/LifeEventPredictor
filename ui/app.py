"""
Wealthsimple Pulse -- Financial Curator Dashboard

Human-in-the-Loop traceability UI for reviewing AI-staged product recommendations.
Displays: Queue View, Traceability Panel (Spending Buffer, Target Product Yield,
Audit Log), and Approve/Reject flow.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.classifier.persona_classifier import classify_persona_tier
from src.features.pipeline import build_features
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

# ---------------------------------------------------------------------------
# Metric definitions (business + technical) for info icons
# ---------------------------------------------------------------------------
METRIC_DEFINITIONS = {
    "liquid_cash": {
        "label": "Liquid Cash",
        "business": "Estimated immediately accessible cash for the client after accounting for current balances and near-term obligations. Used to assess capacity for new product uptake and liquidity risk.",
        "technical": "Currently approximated as 10% of AUA (Assets Under Administration); configurable. In production would incorporate chequing balance and short-term holdings.",
    },
    "monthly_burn_rate": {
        "label": "Monthly Burn Rate",
        "business": "Total spending (debits) per month from chequing and credit card. Indicates lifestyle cost and how much income is consumed by expenses.",
        "technical": "Sum of negative amounts over the last 30 days for account_type in (chequing, credit_card), excluding rent and CC payment transfers.",
    },
    "months_of_runway": {
        "label": "Months of Runway",
        "business": "How many months the client can sustain their recent burn rate before depleting the estimated liquid buffer. Lower runway suggests caution for illiquid products.",
        "technical": "liquid_cash / monthly_burn_rate (spend_velocity_30d). Clipped when burn rate is zero.",
    },
    "aua_current": {
        "label": "AUA (Assets Under Administration)",
        "business": "Total investable assets held with the institution. Drives persona tier (Aspiring Affluent / Sticky Family Leader / Generation Nerd) and product eligibility.",
        "technical": "Sum of balance_after across all investment account types (RRSP, TFSA, RESP, non-reg) at month-end.",
    },
    "spend_velocity_30d": {
        "label": "Spend Velocity (30d)",
        "business": "Rolling 30-day total spending. Used to gauge burn rate and savings capacity.",
        "technical": "Sum of absolute value of debits over the last 30 days for chequing and credit_card transactions.",
    },
    "savings_rate": {
        "label": "Savings Rate",
        "business": "Proportion of income not spent. High savings rate supports loan repayment and product adoption (e.g. RRSP loan).",
        "technical": "(monthly_income - spend_velocity_30d) / monthly_income, where income is inferred from payroll ACH deposits.",
    },
    "confidence_score": {
        "label": "Confidence Score",
        "business": "Model probability that this user is exhibiting the detected signal. Higher scores indicate stronger evidence; thresholds are tuned for target precision to limit false positives.",
        "technical": "XGBoost predicted probability for the binary signal (e.g. leapfrog_ready) for this persona tier. Per-persona thresholds (e.g. 0.5–0.75) were chosen to achieve ~0.80 precision on holdout.",
    },
    "suggested_amount": {
        "label": "Suggested Amount",
        "business": "Recommended product amount (e.g. RRSP loan size) to achieve the stated outcome (e.g. crossing Premium threshold).",
        "technical": "For Aspiring Affluent: estimated from unused RRSP room and gap to $100k AUA. For other personas, product-specific logic.",
    },
    "mcc_entropy": {
        "label": "MCC Entropy",
        "business": "Diversity of spending categories. Lower entropy can indicate a shift toward goal-related categories (e.g. real estate, education).",
        "technical": "Shannon entropy (base 2) of the distribution of spend by MCC category over the observation window.",
    },
    "illiquidity_ratio": {
        "label": "Illiquidity Ratio",
        "business": "Share of AUA in less liquid investments (e.g. Summit/PE). High ratio with high credit spend can signal liquidity risk.",
        "technical": "Balance in investment_non_reg (or illiquid bucket) / total AUA at month-end.",
    },
    "credit_spend_vs_invest": {
        "label": "Credit Spend vs Invest",
        "business": "Ratio of credit card spending to investment transfers. High ratio with large illiquid allocation may warrant Liquidity Watchdog.",
        "technical": "Monthly credit card debits / monthly internal_transfer inflows to investment accounts.",
    },
    "rrsp_utilization": {
        "label": "RRSP Utilization",
        "business": "How much of available RRSP room has been used. Low utilization with high income supports RRSP loan (Leapfrog) recommendation.",
        "technical": "Cumulative RRSP deposits / (rrsp_room + cumulative RRSP deposits).",
    },
}


def _metric_with_info(metric_key: str, value: str, delta_color: str | None = None, key_suffix: str = ""):
    """Render a metric with an info icon that opens a popover with business and technical definitions."""
    col_metric, col_info = st.columns([5, 1])
    with col_metric:
        if delta_color:
            st.metric(METRIC_DEFINITIONS[metric_key]["label"], value, delta_color=delta_color)
        else:
            st.metric(METRIC_DEFINITIONS[metric_key]["label"], value)
    with col_info:
        with st.popover("ℹ", help="Definition", key=f"pop_{metric_key}_{key_suffix}"):
            st.markdown("**Business**")
            st.caption(METRIC_DEFINITIONS[metric_key]["business"])
            st.markdown("**Technical**")
            st.caption(METRIC_DEFINITIONS[metric_key]["technical"])


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
# Cohort storage
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
        "cohort_id": cohort_id,
        "name": name,
        "created_at": now,
        "filters": json.dumps(filters),
    }])
    cohorts_df = pd.concat([cohorts_df, new_cohort], ignore_index=True)
    members_df = pd.concat([members_df, pd.DataFrame(members)], ignore_index=True)
    write_parquet(cohorts_df, cohorts_path)
    write_parquet(members_df, members_path)


def _cohort_metrics(cohort_id: str, decisions: dict):
    """Compute approval/rejection/pending counts and avg confidence for a cohort."""
    members = _load_cohort_members_df()
    members = members[members["cohort_id"] == cohort_id]
    if members.empty:
        return {"size": 0, "approved": 0, "rejected": 0, "pending": 0, "avg_confidence": 0.0}
    approved = sum(1 for uid in members["user_id"] if decisions.get(uid, {}).get("action") == "approved")
    rejected = sum(1 for uid in members["user_id"] if decisions.get(uid, {}).get("action") == "rejected")
    pending = sum(1 for uid in members["user_id"] if decisions.get(uid, {}).get("action") == "pending")
    uncounted = len(members) - approved - rejected - pending
    avg_conf = members["confidence"].mean()
    return {
        "size": len(members),
        "approved": approved,
        "rejected": rejected,
        "pending": pending,
        "uncounted": uncounted,
        "avg_confidence": round(avg_conf, 3),
        "approval_rate": round(approved / (approved + rejected), 3) if (approved + rejected) > 0 else None,
    }


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
    pending = sum(1 for v in st.session_state.decisions.values() if v["action"] == "pending")
    st.sidebar.metric("Approved", approved)
    st.sidebar.metric("Rejected", rejected)
    st.sidebar.metric("Pending", pending)

    # Generate hypotheses
    hypotheses = generate_hypotheses(features, profiles, model)

    # Apply filters
    filtered = [
        h for h in hypotheses
        if h["persona_tier"] in tier_filter and h["confidence"] >= confidence_min
    ]

    # ---------------------------------------------------------------------------
    # Sidebar: Cohort Builder
    # ---------------------------------------------------------------------------
    st.sidebar.divider()
    st.sidebar.markdown("**Cohort Builder**")
    cohort_name = st.sidebar.text_input("Cohort name", key="cohort_name", placeholder="e.g. High-confidence Aspiring Affluent")
    cohort_tiers = st.sidebar.multiselect(
        "Persona tiers",
        options=[k for k in TIER_LABELS if k != "not_eligible"],
        default=[k for k in TIER_LABELS if k != "not_eligible"],
        format_func=lambda x: TIER_LABELS[x],
        key="cohort_tiers",
    )
    conf_low, conf_high = st.sidebar.slider("Confidence range", 0.0, 1.0, (0.5, 1.0), 0.05, key="cohort_conf")
    cohort_products = st.sidebar.multiselect(
        "Products",
        options=PRODUCT_CODES,
        default=PRODUCT_CODES,
        key="cohort_products",
    )
    provinces = sorted(profiles["province"].dropna().unique().tolist()) if not profiles.empty else []
    cohort_provinces = st.sidebar.multiselect("Provinces (optional)", options=provinces, default=[], key="cohort_provinces")
    age_min = st.sidebar.number_input("Age min (optional)", min_value=18, max_value=100, value=18, key="cohort_age_min")
    age_max = st.sidebar.number_input("Age max (optional)", min_value=18, max_value=100, value=100, key="cohort_age_max")

    if st.sidebar.button("Create Cohort", key="create_cohort"):
        subset = [
            h for h in filtered
            if h["persona_tier"] in cohort_tiers
            and cohort_products and h["traceability"]["target_product"]["code"] in cohort_products
            and conf_low <= h["confidence"] <= conf_high
            and (not cohort_provinces or h.get("province") in cohort_provinces)
            and (h.get("age", 0) >= age_min and h.get("age", 99) <= age_max)
        ]
        if not cohort_name.strip():
            st.sidebar.warning("Enter a cohort name.")
        elif not subset:
            st.sidebar.warning("No hypotheses match the selected filters.")
        else:
            latest_month = features["month"].max() if not features.empty else ""
            cid = str(uuid.uuid4())
            filters = {
                "persona_tiers": cohort_tiers,
                "confidence_range": [conf_low, conf_high],
                "products": cohort_products,
                "provinces": cohort_provinces or None,
                "age_range": [age_min, age_max],
            }
            members = [
                {
                    "cohort_id": cid,
                    "user_id": h["user_id"],
                    "persona_tier": h["persona_tier"],
                    "signal": h["signal"],
                    "confidence": float(h["confidence"]),
                    "product_code": h["traceability"]["target_product"]["code"],
                    "snapshot_month": str(latest_month),
                    "age": h.get("age"),
                    "province": h.get("province"),
                }
                for h in subset
            ]
            _save_cohort(cid, cohort_name.strip(), filters, members)
            st.sidebar.success(f"Cohort '{cohort_name.strip()}' created with {len(members)} members.")

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

    # ---------------------------------------------------------------------------
    # Cohort Explorer
    # ---------------------------------------------------------------------------
    with st.expander("Cohort Explorer", expanded=False):
        cohorts_df = _load_cohorts_df()
        if cohorts_df.empty:
            st.caption("No cohorts yet. Create one from the sidebar Cohort Builder.")
        else:
            cohort_names = cohorts_df["name"].tolist()
            selected_name = st.selectbox("Select cohort", options=cohort_names, key="cohort_select")
            if selected_name:
                row = cohorts_df[cohorts_df["name"] == selected_name].iloc[0]
                cid = row["cohort_id"]
                metrics = _cohort_metrics(cid, st.session_state.decisions)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Size", metrics["size"])
                m2.metric("Approved", metrics["approved"])
                m3.metric("Rejected", metrics["rejected"])
                m4.metric("Pending", metrics["pending"])
                st.caption(f"Avg confidence: {metrics['avg_confidence']:.2f}" + (f"  |  Approval rate: {metrics['approval_rate']:.1%}" if metrics.get("approval_rate") is not None else ""))
                members_df = _load_cohort_members_df()
                members_df = members_df[members_df["cohort_id"] == cid][["user_id", "persona_tier", "signal", "confidence", "product_code", "age", "province"]]
                members_df["user_id"] = members_df["user_id"].str[:12] + "..."
                st.dataframe(members_df.head(20), use_container_width=True, hide_index=True)

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
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"**Confidence Score**")
        with c2:
            with st.popover("ℹ", help="What this score means", key="pop_confidence_header"):
                st.markdown("**Business**")
                st.caption(METRIC_DEFINITIONS["confidence_score"]["business"])
                st.markdown("**Technical**")
                st.caption(METRIC_DEFINITIONS["confidence_score"]["technical"])
        _render_confidence_gauge(hypothesis["confidence"])

    with st.expander("About Confidence Score", expanded=False):
        st.markdown(
            "The confidence score is the **model predicted probability** that this user is exhibiting the detected signal "
            "(e.g. leapfrog_ready, liquidity_warning, harvest_opportunity). Per-persona binary classifiers (XGBoost) "
            "output a probability; we apply a threshold tuned to achieve **target precision (e.g. ≥ 0.80)** on holdout data, "
            "so that fewer false positives are surfaced to curators."
        )
        st.markdown("**Interpretation by range:**")
        st.markdown(
            "| Range | Band | Meaning |\n"
            "|-------|------|--------|\n"
            "| 0.90 – 1.00 | High | Very strong signal; historically high precision in backtests. |\n"
            "| 0.75 – 0.90 | High | Strong signal; good candidate for approval but worth a quick review. |\n"
            "| 0.60 – 0.75 | Medium | Moderate signal; use additional judgment and context. |\n"
            "| &lt; 0.60 | Low | Weak signal; typically not surfaced unless part of exploratory experiments. |"
        )

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
            runway_icon = "normal"
        elif runway >= 3:
            runway_icon = "off"
        else:
            runway_icon = "inverse"

        _metric_with_info("liquid_cash", f"${sb['liquid_cash']:,.0f}", key_suffix="sb1")
        _metric_with_info("monthly_burn_rate", f"${sb['monthly_burn_rate']:,.0f}", key_suffix="sb2")
        _metric_with_info("months_of_runway", f"{runway:.1f}", delta_color=runway_icon, key_suffix="sb3")

    # --- Target Product Yield ---
    with prod_col:
        st.markdown("### Target Product")
        tp = trace["target_product"]
        st.markdown(f"**{tp['name']}**")
        st.caption(f"Code: `{tp['code']}`")
        if tp.get("projected_yield"):
            st.success(f"Projected Yield: {tp['projected_yield']}")
        if tp.get("suggested_amount"):
            _metric_with_info("suggested_amount", f"${tp['suggested_amount']:,.0f}", key_suffix="tp1")

    # --- Audit Log ---
    with audit_col:
        st.markdown("### Audit Log")
        with st.popover("ℹ Feature definitions", help="Business and technical definitions for features in the audit log", key="pop_audit_defs"):
            for key in ["aua_current", "mcc_entropy", "illiquidity_ratio", "credit_spend_vs_invest", "rrsp_utilization", "savings_rate", "spend_velocity_30d"]:
                if key in METRIC_DEFINITIONS:
                    st.markdown(f"**{METRIC_DEFINITIONS[key]['label']}**")
                    st.caption(METRIC_DEFINITIONS[key]["business"])
                    st.caption(METRIC_DEFINITIONS[key]["technical"])
                    st.divider()
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
    # Action Bar -- Reject (left) | Pending | Approve (right)
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("Curator Decision")

    existing = st.session_state.decisions.get(user_id, {})
    is_approved = existing.get("action") == "approved"
    is_rejected = existing.get("action") == "rejected"

    # Columns: Reject (left), Pending (center), Approve (right), then reason/status
    reject_col, pending_col, approve_col, reason_col = st.columns([1, 1, 1, 2])

    with reject_col:
        st.markdown('<span style="color: #e74c3c; font-weight: bold;">Reject</span>', unsafe_allow_html=True)
        if st.button(
            "Reject",
            type="secondary",
            use_container_width=True,
            disabled=is_rejected or is_approved,
            key="btn_reject",
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

    with pending_col:
        st.markdown("**Pending / Reconsider**")
        if st.button(
            "Mark Pending",
            use_container_width=True,
            disabled=is_approved or is_rejected,
            key="btn_pending",
        ):
            st.session_state.decisions[user_id] = {
                "action": "pending",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": hypothesis["signal"],
                "persona_tier": hypothesis["persona_tier"],
                "confidence": hypothesis["confidence"],
            }
            st.rerun()

    with approve_col:
        st.markdown('<span style="color: #27ae60; font-weight: bold;">Approve</span>', unsafe_allow_html=True)
        if st.button(
            "Approve",
            type="primary",
            use_container_width=True,
            disabled=is_approved or is_rejected,
            key="btn_approve",
        ):
            st.session_state.decisions[user_id] = {
                "action": "approved",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": hypothesis["signal"],
                "persona_tier": hypothesis["persona_tier"],
                "confidence": hypothesis["confidence"],
            }
            st.rerun()

    with reason_col:
        if is_rejected:
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
        elif existing["action"] == "pending":
            st.info(f"Pending review since {existing.get('timestamp', 'N/A')}. You may still Approve or Reject.")


def _confidence_band(confidence: float) -> tuple[str, str]:
    """Return (band_name, short_description) for the confidence value."""
    if confidence >= 0.90:
        return "High", "Very strong signal; historically high precision in backtests."
    if confidence >= 0.75:
        return "High", "Strong signal; good candidate for approval but worth a quick review."
    if confidence >= 0.60:
        return "Medium", "Moderate signal; use additional judgment and context."
    return "Low", "Weak signal; typically not surfaced unless part of exploratory experiments."


def _render_confidence_gauge(confidence: float):
    """Render a confidence score as a colored metric with band label and optional About expander."""
    if confidence >= 0.8:
        color = "normal"
    elif confidence >= 0.6:
        color = "off"
    else:
        color = "inverse"
    st.metric("Score", f"{confidence:.1%}", delta_color=color)
    band_name, band_desc = _confidence_band(confidence)
    band_note = " (target precision ≥ 0.80)" if band_name == "High" and confidence >= 0.8 else ""
    st.caption(f"Band: **{band_name}**{band_note}")
    st.caption(band_desc)


if __name__ == "__main__":
    main()
