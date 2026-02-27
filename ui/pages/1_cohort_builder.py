"""
Cohort Builder page: cohort name, filters, Create Cohort, Cohort Explorer, Cluster Map, Scenario Planning.
"""

import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.lib import (
    TIER_LABELS,
    PRODUCT_CODES,
    get_default_macro,
    load_data,
    load_model,
    generate_hypotheses,
    load_cohorts_df,
    save_cohort,
    cohort_metrics,
)
from src.api.macro_agent import MacroSnapshot

st.set_page_config(page_title="Cohort Builder — Wealthsimple Pulse", page_icon="W", layout="wide")

# Session state for decisions (shared with main app)
if "decisions" not in st.session_state:
    st.session_state.decisions = {}
if "macro" not in st.session_state:
    st.session_state.macro = get_default_macro()

# Back to Dashboard
if st.button("Back to Dashboard"):
    st.switch_page("ui/app.py")

st.title("Cohort Builder")
st.caption("Create and explore cohorts by persona, confidence, and product. Use the cluster map to define custom ranges.")

# Scenario Planning (shared macro)
with st.expander("Scenario Planning", expanded=False):
    macro = st.session_state.macro
    boc_rate = st.slider("BoC Prime Rate (%)", 3.0, 8.0, float(macro.boc_prime_rate), 0.25, key="cb_boc")
    vix_val = st.slider("Market Volatility (VIX)", 10, 40, int(macro.vix), 1, key="cb_vix")
    st.session_state.macro = MacroSnapshot(boc_prime_rate=boc_rate, vix=vix_val)
    st.caption(f"Current scenario: BoC {boc_rate}%, VIX {vix_val}. Hypotheses below use this scenario.")

macro = st.session_state.macro
profiles, txns, features = load_data()
model = load_model()
hypotheses = generate_hypotheses(features, profiles, model, macro)

# Cohort name and filters
st.subheader("Create cohort")
col_name, col_filters = st.columns([1, 2])
with col_name:
    cohort_name = st.text_input("Cohort name", placeholder="e.g. High-conf Aspiring Affluent", key="cb_name")
with col_filters:
    cohort_tiers = st.multiselect(
        "Persona tiers",
        options=[k for k in TIER_LABELS if k != "not_eligible"],
        default=[k for k in TIER_LABELS if k != "not_eligible"],
        format_func=lambda x: TIER_LABELS[x],
        key="cb_tiers",
    )
    conf_low, conf_high = st.slider("Confidence range", 0.0, 1.0, (0.5, 1.0), 0.05, key="cb_conf")
    cohort_products = st.multiselect("Products", options=PRODUCT_CODES, default=PRODUCT_CODES, key="cb_prod")

filtered = [
    h for h in hypotheses
    if h["persona_tier"] in cohort_tiers
    and h["traceability"]["target_product"]["code"] in cohort_products
    and conf_low <= h["confidence"] <= conf_high
]
if macro.boc_prime_rate > 6.0:
    filtered = [h for h in filtered if h["traceability"]["target_product"]["code"] != "RRSP_LOAN"]

if st.button("Create Cohort"):
    if not cohort_name.strip():
        st.warning("Enter a cohort name.")
    elif not filtered:
        st.warning("No hypotheses match filters.")
    else:
        cid = str(uuid.uuid4())
        latest_month = features["month"].max() if not features.empty else ""
        save_cohort(cid, cohort_name.strip(), {
            "persona_tiers": cohort_tiers,
            "confidence_range": [conf_low, conf_high],
            "products": cohort_products,
        }, [
            {
                "cohort_id": cid, "user_id": h["user_id"], "persona_tier": h["persona_tier"],
                "signal": h["signal"], "confidence": float(h["confidence"]),
                "product_code": h["traceability"]["target_product"]["code"],
                "snapshot_month": str(latest_month), "age": h.get("age"), "province": h.get("province"),
            } for h in filtered
        ])
        st.success(f"Cohort '{cohort_name.strip()}' created ({len(filtered)} members).")
        st.rerun()

st.divider()

# Cohort Explorer
st.subheader("Cohort Explorer")
cohorts_df = load_cohorts_df()
if cohorts_df.empty:
    st.caption("No cohorts yet. Create one above.")
else:
    selected_name = st.selectbox("Select cohort", options=cohorts_df["name"].tolist(), key="cb_select")
    if selected_name:
        crow = cohorts_df[cohorts_df["name"] == selected_name].iloc[0]
        metrics = cohort_metrics(crow["cohort_id"], st.session_state.decisions)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Size", metrics["size"])
        m2.metric("Approved", metrics["approved"])
        m3.metric("Rejected", metrics["rejected"])
        m4.metric("Pending", metrics["pending"])

st.divider()

# Cluster Map — Custom chart
st.subheader("Cluster Map — Custom Cohort")
if filtered:
    map_data = []
    for h in filtered:
        sb = h.get("traceability", {}).get("spending_buffer", {})
        map_data.append({
            "user_id": h["user_id"][:8],
            "Spend Velocity (30d)": sb.get("monthly_burn_rate", 0),
            "Months Runway": sb.get("months_of_runway", 0),
            "Persona": TIER_LABELS.get(h["persona_tier"], h["persona_tier"]),
        })
    map_df = pd.DataFrame(map_data)
    if not map_df.empty:
        fig = px.scatter(
            map_df,
            x="Spend Velocity (30d)",
            y="Months Runway",
            color="Persona",
            hover_data=["user_id"],
            title="Users by Spend Velocity vs Liquidity Runway",
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        fig.update_layout(showlegend=True, height=400)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        st.plotly_chart(fig, use_container_width=True)
        vel_max_val = float(map_df["Spend Velocity (30d)"].max() + 1)
        vel_min, vel_max = st.slider("Spend velocity range", 0.0, vel_max_val, (0.0, vel_max_val), key="cb_vel")
        run_min, run_max = st.slider("Runway range (months)", 0.0, 24.0, (0.0, 24.0), key="cb_run")
        custom = [
            h for h in filtered
            if vel_min <= h["traceability"]["spending_buffer"]["monthly_burn_rate"] <= vel_max
            and run_min <= h["traceability"]["spending_buffer"]["months_of_runway"] <= run_max
        ]
        st.caption(f"Custom cohort: {len(custom)} users in selected range.")
else:
    st.info("No hypotheses match the current filters. Adjust persona tiers, confidence range, or products above.")
