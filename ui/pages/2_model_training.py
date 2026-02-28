"""
Model Training & Evaluation page: data generation summary, per-persona metrics,
feature importance, and production/drift considerations.
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.io import DATA_RAW, DATA_PROCESSED, load_config, read_parquet
from ui.lib import TIER_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "data"
PERSONAS = ["aspiring_affluent", "sticky_family_leader", "generation_nerd"]

st.set_page_config(
    page_title="Model Training — Wealthsimple Pulse",
    page_icon="W",
    layout="wide",
)

if st.button("Back to Dashboard"):
    st.switch_page("ui/app.py")

st.title("Model Training & Evaluation")
st.caption(
    "Data generation process, per-persona XGBoost performance, global feature importance, "
    "and production considerations."
)

# ---------------------------------------------------------------------------
# 1. Data generation process
# ---------------------------------------------------------------------------
st.header("1. Data generation process")

config = load_config()
gen = config.get("data_generation", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", gen.get("num_users", "—"))
c2.metric("Months", gen.get("months", "—"))
c3.metric("Start date", gen.get("start_date", "—"))
c4.metric("Seed", gen.get("seed", "—"))

st.subheader("Persona mix")
weights = gen.get("persona_weights", {})
if weights:
    w_cols = st.columns(len(weights))
    for i, (k, v) in enumerate(weights.items()):
        with w_cols[i]:
            st.metric(k.replace("_", " ").title(), f"{float(v):.0%}")

st.subheader("Ranges")
aua = gen.get("aua_ranges", {})
income = gen.get("income_brackets", {})
if aua or income:
    left, right = st.columns(2)
    with left:
        if aua:
            st.markdown("**AUA ranges (CAD)**")
            for k, v in aua.items():
                st.caption(f"{k}: {v}")
    with right:
        if income:
            st.markdown("**Income brackets (CAD)**")
            for k, v in income.items():
                st.caption(f"{k}: {v}")

if gen.get("signal_onset_buffer_months") is not None:
    st.caption(f"Signal onset buffer: {gen['signal_onset_buffer_months']} months")
if gen.get("txns_per_day_lambda") is not None:
    st.caption(f"Transactions per day (lambda): {gen['txns_per_day_lambda']}")

# Dataset stats
profiles_path = DATA_RAW / "user_profiles.parquet"
txns_path = DATA_RAW / "transactions.parquet"
if profiles_path.exists() and txns_path.exists():
    try:
        profiles_df = read_parquet(profiles_path)
        txns_df = read_parquet(txns_path)
        if "timestamp" in txns_df.columns:
            txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])
        st.subheader("Dataset stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Profiles", len(profiles_df))
        col2.metric("Transactions", f"{len(txns_df):,}")
        if not txns_df.empty and "timestamp" in txns_df.columns:
            col3.metric("Date range", f"{txns_df['timestamp'].min().date()} → {txns_df['timestamp'].max().date()}")
    except Exception as e:
        st.caption(f"Could not load dataset stats: {e}")

st.divider()

# ---------------------------------------------------------------------------
# 2. Per-persona model performance
# ---------------------------------------------------------------------------
st.header("2. Per-persona model performance")

artifacts = {}
for persona in PERSONAS:
    path = MODEL_DIR / f"model_{persona}.joblib"
    if path.exists():
        try:
            artifacts[persona] = joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load model for {persona}: {e}")
    else:
        st.info(f"Model not found for **{TIER_LABELS.get(persona, persona)}**. Run `python -m src.models.train`.")

for persona in PERSONAS:
    if persona not in artifacts:
        continue
    art = artifacts[persona]
    display_name = TIER_LABELS.get(persona, persona)
    st.subheader(display_name)
    target_label = art.get("target_label", "")
    st.caption(f"Target signal: **{target_label}**")
    metrics = art.get("metrics", {})
    if metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        m2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        m3.metric("F1", f"{metrics.get('f1', 0):.3f}")
        m4.metric("AUC-ROC", f"{metrics.get('auc', 0):.3f}")
        m5.metric("Optimal threshold", f"{metrics.get('optimal_threshold', 0.5):.3f}")
    if art.get("classification_report"):
        with st.expander("Classification report"):
            st.text(art["classification_report"])
    st.divider()

# ---------------------------------------------------------------------------
# 3. Global feature importance
# ---------------------------------------------------------------------------
st.header("3. Global feature importance")

for persona in PERSONAS:
    if persona not in artifacts:
        continue
    art = artifacts[persona]
    model = art.get("model")
    feat_names = art.get("features", [])
    if model is None or not feat_names or not hasattr(model, "feature_importances_"):
        continue
    display_name = TIER_LABELS.get(persona, persona)
    st.subheader(display_name)
    importances = model.feature_importances_
    pairs = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    names = [p[0] for p in pairs]
    vals = [float(p[1]) for p in pairs]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(color="#000000", line=dict(width=0)),
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=120),
        xaxis_title="Importance",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 4. Production considerations and context drift
# ---------------------------------------------------------------------------
st.header("4. Production considerations and context drift")

with st.expander("Context drift and what to watch", expanded=True):
    st.markdown("""
    **Possible signs of context drift:**
    - **Feature distribution shift:** Mean or median of key features (e.g. spend velocity, AUA, savings rate) in the recent inference window differs from the training period.
    - **Drop in precision or approval rate:** For a given persona or product, precision or curator approval rate declines over time.
    - **Change in mix:** Persona mix or signal prevalence in production no longer matches the training data (e.g. fewer Leapfrog signals than expected).
    """)

with st.expander("Current feature summary (latest month)", expanded=False):
    features_path = DATA_PROCESSED / "features.parquet"
    if features_path.exists():
        try:
            feats = read_parquet(features_path)
            if "month" in feats.columns and not feats.empty:
                latest = feats["month"].max()
                latest_df = feats[feats["month"] == latest]
                numeric = latest_df.select_dtypes(include=["number"]).columns.tolist()
                if numeric:
                    summary = latest_df[numeric].agg(["mean", "std", "min", "max"]).T
                    summary.columns = ["Mean", "Std", "Min", "Max"]
                    st.dataframe(summary.round(4), use_container_width=True, hide_index=True)
                    st.caption(f"Latest month: {latest}. Compare to training-period stats when baseline is logged.")
                else:
                    st.caption("No numeric columns in features.")
            else:
                st.caption("No month column or empty features.")
        except Exception as e:
            st.caption(f"Could not load features: {e}")
    else:
        st.caption("Features file not found. Run the feature pipeline first.")

with st.expander("Recommendations for production", expanded=True):
    st.markdown("""
    - **Log predictions and outcomes:** Store model scores and curator approve/reject decisions per cohort (e.g. in the existing feedback DB) to compute precision/recall over time.
    - **Monitor by segment:** Track precision, recall, and approval rate by persona and product; alert if metrics drop below a threshold.
    - **Retrain cadence:** Define a retrain schedule (e.g. quarterly); trigger retrain if validation metrics degrade or feature distributions shift significantly.
    """)
