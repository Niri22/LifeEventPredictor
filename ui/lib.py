"""
Shared UI helpers, constants, data/model loading, rendering primitives
used across Control Center, Decision Console, and Growth Engine pages.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.api.feedback import apply_feedback_penalty, get_feedback_stats, record_feedback
from src.api.macro_agent import MacroSnapshot, adjust_confidence_for_macro
from src.classifier.guardrails import apply_guardrails_to_hypothesis
from src.classifier.cohort_engine import build_intent_cohorts
from src.classifier.persona_classifier import classify_persona_tier
from src.features.nudges import generate_composite_reason, generate_nudge
from src.models.governance import enrich_hypothesis_with_governance
from src.models.predict import XGBSignalModel, predict_signal
from src.utils.io import DATA_RAW, DATA_PROCESSED, DATA_EXPERIMENTS, read_parquet, write_parquet, load_config

# ---------------------------------------------------------------------------
# Palette tokens
# ---------------------------------------------------------------------------
COIN_WHITE = "#FFFFFF"
MIDNIGHT = "#000000"
WS_GOLD = "#FFB547"
SAGE_GREEN = "#E8F0E8"
STONE_GREY = "#F2F2F2"

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------
TIER_LABELS = {
    "aspiring_affluent": "Momentum Builder ($50k-$100k)",
    "sticky_family_leader": "Full-Stack Client ($100k-$500k)",
    "generation_nerd": "Legacy Architect ($500k+)",
    "not_eligible": "Not Eligible (<$50k)",
}

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

PRODUCT_CODES = ["RRSP_LOAN", "SUMMIT_PORTFOLIO", "AI_RESEARCH_DIRECT_INDEX"]

COHORT_COLUMNS = ["cohort_id", "name", "created_at", "filters"]
MEMBER_COLUMNS = [
    "cohort_id", "user_id", "persona_tier", "signal", "confidence",
    "product_code", "snapshot_month", "age", "province",
]

DEFAULT_BOC_RATE = 4.25
DEFAULT_VIX = 18.0

METRIC_DEFINITIONS = {
    "liquid_cash": {
        "label": "Liquid Cash",
        "business": "Estimated immediately accessible cash for the client.",
        "technical": "Approximated as 10% of AUA; would incorporate chequing balance in production.",
    },
    "monthly_burn_rate": {
        "label": "Monthly Burn Rate",
        "business": "Total spending (debits) per month from chequing and credit card.",
        "technical": "Sum of negative amounts over the last 30 days for chequing + credit_card.",
    },
    "months_of_runway": {
        "label": "Months of Runway",
        "business": "How many months the client can sustain their recent burn rate.",
        "technical": "liquid_cash / spend_velocity_30d.",
    },
    "confidence_score": {
        "label": "Confidence Score",
        "business": "Model probability that this user is exhibiting the detected signal.",
        "technical": "XGBoost predicted probability, adjusted by macro conditions and active learning feedback.",
    },
    "suggested_amount": {
        "label": "Suggested Amount",
        "business": "Recommended product amount to achieve the stated outcome.",
        "technical": "Estimated from unused RRSP room and gap to $100k AUA.",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data"
PERSONAS = ["aspiring_affluent", "sticky_family_leader", "generation_nerd"]


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


def generate_hypotheses(features: pd.DataFrame, profiles: pd.DataFrame, model: XGBSignalModel, macro: MacroSnapshot):
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

        adj_conf, macro_reasons = adjust_confidence_for_macro(
            result["confidence"], persona_tier, product_code, macro,
        )
        fb_conf, fb_reason = apply_feedback_penalty(
            adj_conf, persona_tier, result["signal"], product_code,
        )
        result["confidence"] = fb_conf
        result["macro_reasons"] = macro_reasons
        result["feedback_reason"] = fb_reason

        feat_for_nudge = {**feat_dict}
        if result.get("traceability", {}).get("target_product", {}).get("suggested_amount"):
            feat_for_nudge["suggested_amount"] = result["traceability"]["target_product"]["suggested_amount"]
        behavioral_nudge = generate_nudge(persona_tier, result["signal"], feat_for_nudge, product_name)
        result["nudge"] = generate_composite_reason(behavioral_nudge, macro_reasons, fb_reason)

        enrich_hypothesis_with_governance(result)
        apply_guardrails_to_hypothesis(result, feat_dict)
        if result.get("life_inflection_alert") and product_code == "RRSP_LOAN":
            continue

        staged_at = datetime.now(timezone.utc).isoformat()
        result["staged_at"] = staged_at
        result["hypothesis_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{result['user_id']}|{result.get('signal', '')}|{product_code}|{staged_at}"))
        hypotheses.append(result)

    return hypotheses


def load_cohorts_df() -> pd.DataFrame:
    path = DATA_EXPERIMENTS / "cohorts.parquet"
    if not path.exists():
        return pd.DataFrame(columns=COHORT_COLUMNS)
    return read_parquet(path)


def load_cohort_members_df() -> pd.DataFrame:
    path = DATA_EXPERIMENTS / "cohort_members.parquet"
    if not path.exists():
        return pd.DataFrame(columns=MEMBER_COLUMNS)
    return read_parquet(path)


def save_cohort(cohort_id: str, name: str, filters: dict, members: list):
    DATA_EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    cohorts_path = DATA_EXPERIMENTS / "cohorts.parquet"
    members_path = DATA_EXPERIMENTS / "cohort_members.parquet"
    cohorts_df = load_cohorts_df()
    members_df = load_cohort_members_df()
    now = datetime.now(timezone.utc).isoformat()
    new_cohort = pd.DataFrame([{
        "cohort_id": cohort_id, "name": name, "created_at": now,
        "filters": json.dumps(filters),
    }])
    cohorts_df = pd.concat([cohorts_df, new_cohort], ignore_index=True)
    members_df = pd.concat([members_df, pd.DataFrame(members)], ignore_index=True)
    write_parquet(cohorts_df, cohorts_path)
    write_parquet(members_df, members_path)


def cohort_metrics(cohort_id: str, decisions: dict) -> dict:
    members = load_cohort_members_df()
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


def get_default_macro() -> MacroSnapshot:
    return MacroSnapshot(boc_prime_rate=DEFAULT_BOC_RATE, vix=DEFAULT_VIX)


def get_experiment_metrics():
    """Load experiment assignments + outcomes and compute pathway metrics (no API required)."""
    try:
        from src.experiments.storage import read_assignments, read_outcomes
        from src.experiments.metrics import compute_pathway_metrics
        config = load_config()
        assignments_df = read_assignments()
        outcomes_df = read_outcomes()
        return compute_pathway_metrics(assignments_df, outcomes_df, config)
    except Exception:
        return pd.DataFrame()


def apply_experiment_reweight(hypothesis: dict):
    """Return (priority_score, safety_actions) for a hypothesis using pathway uplift. Works offline."""
    try:
        from src.experiments.reweight import apply_uplift_reweighting
        config = load_config()
        metrics_df = get_experiment_metrics()
        return apply_uplift_reweighting(hypothesis, metrics_df, config)
    except Exception:
        return (float(hypothesis.get("confidence", 0)), [])


# Experiment Performance display helpers (executive summary, ranked table, labels)
EXPERIMENT_PERSONA_SHORT = {
    "aspiring_affluent": "Momentum Builder",
    "sticky_family_leader": "Full-Stack Client",
    "generation_nerd": "Legacy Architect",
}
EXPERIMENT_PRODUCT_SHORT = {
    "RRSP_LOAN": "RRSP Loan",
    "SUMMIT_PORTFOLIO": "Summit Portfolio",
    "AI_RESEARCH_DIRECT_INDEX": "AI Research + Direct Index",
}


def experiment_persona_label(tier: str) -> str:
    return EXPERIMENT_PERSONA_SHORT.get(tier, tier.replace("_", " ").title())


def experiment_product_label(code: str) -> str:
    return EXPERIMENT_PRODUCT_SHORT.get(code, code)


def get_experiment_summary(metrics_df: pd.DataFrame) -> dict:
    """Compute net uplift, top row, suppressed count, projected AUA for executive summary."""
    if metrics_df.empty:
        return {"net_uplift": 0, "top_row": None, "n_suppressed_sig": 0, "projected_aua": 0}
    sorted_df = metrics_df.sort_values("uplift_score", ascending=False)
    total_n = (sorted_df["n_treatment"] + sorted_df["n_control"]).sum()
    net_uplift = (sorted_df["uplift_score"] * (sorted_df["n_treatment"] + sorted_df["n_control"])).sum() / max(total_n, 1)
    suppressed = sorted_df[sorted_df["uplift_score"] < 0]
    n_suppressed_sig = len(suppressed[suppressed["significance_flag"] == True])
    projected_aua = (sorted_df["delta_aua_uplift"] * (sorted_df["n_treatment"] + sorted_df["n_control"])).sum()
    return {
        "net_uplift": net_uplift,
        "top_row": sorted_df.iloc[0],
        "n_suppressed_sig": n_suppressed_sig,
        "projected_aua": projected_aua,
        "sorted_df": sorted_df.reset_index(drop=True),
    }


def build_ranked_experiment_table(metrics_df: pd.DataFrame):
    """Build ranked df with Persona, Product, Uplift %, AUA Impact, Significance, Status; return (ranked_df, styled_df)."""
    if metrics_df.empty:
        return pd.DataFrame(), None
    summary = get_experiment_summary(metrics_df)
    ranked = summary["sorted_df"].copy()
    ranked["Persona"] = ranked["persona_tier"].map(experiment_persona_label)
    ranked["Product"] = ranked["product_code"].map(experiment_product_label)
    ranked["Uplift %"] = (ranked["uplift_score"] * 100).round(2)
    ranked["AUA Impact"] = ranked["delta_aua_uplift"].round(0)
    ranked["Significance"] = ranked["significance_flag"].apply(lambda x: "Yes" if x else "No")

    def status_badge(row):
        u, sig = float(row["uplift_score"]), row.get("significance_flag", False)
        if sig and u > 0:
            return "Boost"
        if sig and u < 0:
            return "Suppress"
        return "Insufficient data"

    ranked["Status"] = ranked.apply(status_badge, axis=1)
    display_cols = ["Persona", "Product", "Uplift %", "AUA Impact", "Significance", "Status"]
    table_df = ranked[display_cols]

    def row_bg(row):
        s = row["Status"]
        bg = "#E8F5E9" if s == "Boost" else "#FFEBEE" if s == "Suppress" else "#f5f5f5"
        return [f"background-color: {bg};"] * len(row)

    styled = table_df.style.apply(row_bg, axis=1).format(
        {"Uplift %": "{:+.2f}", "AUA Impact": "${:,.0f}"}
    )
    return ranked, styled


# ---------------------------------------------------------------------------
# Rendering primitives (shared across pages)
# ---------------------------------------------------------------------------
def inject_ws_theme():
    """Inject global CSS for Wealthsimple-like visual styling."""
    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');
    :root {{ --ws-midnight: {MIDNIGHT}; --ws-off-white: {COIN_WHITE}; --ws-gold: {WS_GOLD}; --ws-sage: {SAGE_GREEN}; --ws-stone: {STONE_GREY}; --ws-radius: 8px; }}
    .stApp {{ background-color: var(--ws-off-white); color: var(--ws-midnight); font-family: 'Inter', system-ui, sans-serif; }}
    section[data-testid="stSidebar"] > div {{ background-color: var(--ws-stone); }}
    h1, h2 {{ font-family: 'Playfair Display', serif; letter-spacing: 0.01em; }}
    h3, h4, h5, h6, .stMarkdown, .stDataFrame, .stMetric, .stButton > button {{ font-family: 'Inter', system-ui, sans-serif; }}
    .ws-main {{ max-width: 1200px; margin: 0 auto; padding: 1.5rem 2rem 2.5rem 2rem; }}
    .ws-card {{ background: var(--ws-off-white); border-radius: var(--ws-radius); border: 1px solid var(--ws-stone); box-shadow: 0 4px 20px rgba(0,0,0,0.05); padding: 1rem 1.25rem; margin-bottom: 1rem; }}
    .stButton > button {{ border-radius: var(--ws-radius); border: 1px solid transparent; font-weight: 500; padding: 0.35rem 0.9rem; transition: all 0.15s ease-out; }}
    .stButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
    .ws-btn-primary button {{ background-color: var(--ws-midnight); color: var(--ws-off-white); }}
    .ws-btn-danger button {{ background-color: #FDEDEC; color: #C0392B; border-color: #F5B7B1; }}
    .ws-btn-secondary button {{ background-color: var(--ws-stone); color: var(--ws-midnight); }}
    .stMetric > div:first-child {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .stMetric {{ padding: 0.5rem 0; }}
    .ws-alert {{ border-radius: var(--ws-radius); padding: 0.75rem 1rem; margin: 0.4rem 0; border-left: 4px solid; }}
    .ws-alert-red {{ background: #FFEBEE; border-color: #c0392b; }}
    .ws-alert-green {{ background: #E8F5E9; border-color: #0d7d0d; }}
    .ws-alert-amber {{ background: #FFF8E1; border-color: #FFB547; }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def metric_with_info(metric_key: str, value: str, delta_color: str | None = None):
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


def confidence_band(confidence: float) -> tuple[str, str]:
    if confidence >= 0.90:
        return "High", "Very strong signal; historically high precision."
    if confidence >= 0.75:
        return "High", "Strong signal; good candidate for approval."
    if confidence >= 0.60:
        return "Medium", "Moderate signal; use additional judgment."
    return "Low", "Weak signal; exploratory only."


def render_confidence_gauge(confidence: float):
    color = "normal" if confidence >= 0.8 else ("off" if confidence >= 0.6 else "inverse")
    st.metric("Score", f"{confidence:.1%}", delta_color=color)
    band_name, band_desc = confidence_band(confidence)
    st.caption(f"Band: **{band_name}**")
    st.caption(band_desc)


def build_queue_df(items: list) -> pd.DataFrame:
    rows = []
    for i, h in enumerate(items):
        status = st.session_state.decisions.get(h["user_id"], {}).get("action", "pending")
        gov = h.get("governance", {})
        dist = h.get("distance_to_upgrade") or {}
        rows.append({
            "idx": i,
            "User ID": h["user_id"][:12] + "...",
            "Tier": TIER_LABELS.get(h["persona_tier"], h["persona_tier"]),
            "Path": dist.get("cohort_label", ""),
            "Signal": SIGNAL_LABELS.get(h["signal"], h["signal"]),
            "Confidence": h["confidence"],
            "Gov": f"{GOV_TIER_ICONS.get(gov.get('tier', ''), '')} {gov.get('tier', '').upper()}",
            "Product": h["traceability"]["target_product"]["name"],
            "Status": status.upper(),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Model artifact loading (for Growth Engine)
# ---------------------------------------------------------------------------
def load_model_artifacts() -> dict:
    """Load per-persona joblib artifacts. Returns {persona: artifact_dict}."""
    artifacts = {}
    for persona in PERSONAS:
        path = MODEL_DIR / f"model_{persona}.joblib"
        if path.exists():
            artifacts[persona] = joblib.load(path)
    return artifacts


def get_model_reliability_table(artifacts: dict, precision_target: float = 0.80) -> pd.DataFrame:
    """Build compact precision/recall/F1 table with drift flag."""
    rows = []
    for persona in PERSONAS:
        art = artifacts.get(persona)
        if art is None:
            rows.append({"Persona": experiment_persona_label(persona), "Precision": None, "Recall": None, "F1": None, "AUC": None, "Drift Alert": "Model missing"})
            continue
        m = art.get("metrics", {})
        prec = m.get("precision", 0)
        drift = "Below target" if prec < precision_target else "OK"
        rows.append({
            "Persona": experiment_persona_label(persona),
            "Precision": round(prec, 3),
            "Recall": round(m.get("recall", 0), 3),
            "F1": round(m.get("f1", 0), 3),
            "AUC": round(m.get("auc", 0), 3),
            "Drift Alert": drift,
        })
    return pd.DataFrame(rows)
