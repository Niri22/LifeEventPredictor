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

# Consistent terminology
ITEM_TERMINOLOGY = "Cases"  # Use "Cases" consistently instead of "Signals" or "Items"

TIER_COLORS = {
    "aspiring_affluent": "#4ECDC4",
    "sticky_family_leader": "#FFD93D",
    "generation_nerd": "#6C5CE7",
    "not_eligible": "#636E72",
}

SIGNAL_LABELS = {
    "leapfrog_ready": "Leapfrog Readiness",
    "liquidity_warning": "Liquidity Watchdog",
    "harvest_opportunity": "Analyst-in-Pocket",
}

# Persona and signal explanations (for onboarding, tooltips, README)
PERSONA_DESCRIPTIONS = {
    "aspiring_affluent": {
        "name": "Momentum Builder",
        "range": "$50k–$100k AUA",
        "who": "Pre-Premium clients with strong growth velocity.",
        "pain_point": "High ambition, but friction crossing milestone tiers.",
        "signal": "Leapfrog Readiness",
        "action": "Stage RRSP Loan to accelerate Premium conversion.",
        "product": "Retirement Accelerator (RRSP Loan).",
        "example": "$82k AUA + high savings velocity → loan-sized bridge proposed.",
    },
    "sticky_family_leader": {
        "name": "Full-Stack Client",
        "range": "$100k–$500k AUA",
        "who": "Premium clients with multiple goals; core of the business.",
        "pain_point": "Multi-account friction; risk of over-allocating to illiquid exposure.",
        "signal": "Liquidity Watchdog",
        "action": "Monitor allocation; suggest rebalance + visibility tools.",
        "product": "Summit + WS Credit Card.",
        "example": "High transfers + rising credit spend → Suggest rebalance + WS Credit Card.",
    },
    "generation_nerd": {
        "name": "Legacy Architect",
        "range": "$500k+ AUA",
        "who": "High-net-worth clients focused on long-term, multi-generational wealth.",
        "pain_point": "Sophisticated but time-poor; want institutional-grade exposure without high friction.",
        "signal": "Analyst-in-Pocket",
        "action": "Tax-aware optimization and institutional-grade exposure.",
        "product": "Advanced allocation strategies.",
        "example": "Direct index + elevated volatility → Research summary or tax-loss harvest move.",
    },
}

SIGNAL_DESCRIPTIONS = {
    "leapfrog_ready": {
        "label": "Leapfrog Readiness",
        "meaning": "Detected signal: unused RRSP room and savings behavior that supports a loan-sized bridge to Premium.",
        "example": "$82k AUA + high savings velocity → loan-sized bridge proposed.",
    },
    "liquidity_warning": {
        "label": "Liquidity Watchdog",
        "meaning": "Detected signal: transfer and spend patterns that risk over-allocation to illiquid exposure.",
        "example": "High transfers + rising credit spend → Suggest rebalance + WS Credit Card.",
    },
    "harvest_opportunity": {
        "label": "Analyst-in-Pocket",
        "meaning": "Detected signal: demand for research and tax-aware optimization on holdings.",
        "example": "Direct index + elevated volatility → Research summary or tax-loss harvest move.",
    },
}

GOV_TIER_ICONS = {"green": "[G]", "amber": "[A]", "red": "[R]"}

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

# Smaller dataset for bootstrap when data is missing (e.g. Streamlit Cloud)
BOOTSTRAP_NUM_USERS = 150


def _ensure_data_exists():
    """If raw or processed parquet files are missing, generate data and run feature pipeline."""
    profiles_path = DATA_RAW / "user_profiles.parquet"
    txns_path = DATA_RAW / "transactions.parquet"
    features_path = DATA_PROCESSED / "features.parquet"
    if profiles_path.exists() and txns_path.exists() and features_path.exists():
        return
    # Generate synthetic data and features so the app can start (e.g. on first deploy)
    config = load_config()
    config = dict(config)
    if "data_generation" not in config:
        config["data_generation"] = {}
    config["data_generation"] = dict(config["data_generation"])
    config["data_generation"]["num_users"] = min(
        config["data_generation"].get("num_users", 1000), BOOTSTRAP_NUM_USERS
    )
    from src.data_generator.generator import generate_dataset
    from src.features.pipeline import run_pipeline
    from src.models.train import train_all_models
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    generate_dataset(config)
    run_pipeline()
    # Train models if any are missing so the app can produce hypotheses
    for p in PERSONAS:
        if not (MODEL_DIR / f"model_{p}.joblib").exists():
            train_all_models(config)
            break


@st.cache_data
def load_data():
    _ensure_data_exists()
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


@st.cache_data(ttl=120)
def get_cached_hypotheses(boc: float, vix: float):
    """Return hypotheses for the given macro (boc, vix). Cached so Control Center loads fast when switching pages.
    Use macro.boc_prime_rate and macro.vix; when user changes Scenario sliders, cache misses and we recompute."""
    macro = MacroSnapshot(boc_prime_rate=boc, vix=vix)
    profiles, _txns, features = load_data()
    model = load_model()
    return generate_hypotheses(features, profiles, model, macro)


HYPOTHESES_JSON = DATA_PROCESSED / "hypotheses.json"


@st.cache_data(ttl=300)
def load_precomputed_hypotheses():
    """Load hypotheses from data/processed/hypotheses.json (prototype mode). No model inference.
    If file is missing, fall back to get_cached_hypotheses with default macro so the app still runs."""
    if HYPOTHESES_JSON.exists():
        try:
            with open(HYPOTHESES_JSON) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    return get_cached_hypotheses(DEFAULT_BOC_RATE, DEFAULT_VIX)


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


@st.cache_data(ttl=60)
def get_experiment_metrics():
    """Load pathway metrics from disk when available (prototype mode); else compute from assignments/outcomes. Cached 60s."""
    try:
        from src.experiments.storage import read_pathway_metrics, read_assignments, read_outcomes
        from src.experiments.metrics import compute_pathway_metrics
        pathway_metrics_df = read_pathway_metrics()
        if pathway_metrics_df is not None and not pathway_metrics_df.empty:
            return pathway_metrics_df
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
        
        /* Color system - disciplined palette */
        :root {{ 
            --ws-midnight: {MIDNIGHT}; 
            --ws-off-white: {COIN_WHITE}; 
            --ws-gold: {WS_GOLD}; 
            --ws-sage: {SAGE_GREEN}; 
            --ws-stone: {STONE_GREY}; 
            --ws-radius: 8px;
            --ws-red: #DC2626;      /* Risk */
            --ws-amber: #F59E0B;    /* Review */
            --ws-green: #059669;    /* Opportunity */
            --ws-primary: {WS_GOLD}; /* Action */
            --ws-muted: #6B7280;    /* Secondary text */
        }}
        
        /* Base app styling - flat, minimal borders */
        .stApp {{ 
            background-color: var(--ws-off-white); 
            color: var(--ws-midnight); 
            font-family: 'Inter', system-ui, sans-serif; 
        }}
        
        /* Hide Streamlit branding and default navigation */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header {{ visibility: hidden; }}
        [data-testid="stSidebarNav"] {{ display: none !important; }}
        
        /* Typography Hierarchy - Strong contrast between levels */
        
        /* Page title: bold, large, high contrast */
        .ws-page-title {{ 
            font-family: 'Playfair Display', serif; 
            font-size: 2.5rem; 
            font-weight: 700; 
            color: var(--ws-midnight); 
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }}
        
        /* Section headers: medium bold */
        .ws-section-header {{ 
            font-family: 'Inter', sans-serif; 
            font-size: 1.25rem; 
            font-weight: 600; 
            color: var(--ws-midnight); 
            margin: 2rem 0 1rem 0;
            letter-spacing: -0.01em;
        }}
        
        /* Subsection headers */
        .ws-subsection {{ 
            font-family: 'Inter', sans-serif; 
            font-size: 1rem; 
            font-weight: 600; 
            color: var(--ws-midnight); 
            margin: 1.5rem 0 0.75rem 0;
        }}
        
        /* KPI labels: subtle, uppercase, smaller */
        .ws-kpi-label {{ 
            font-size: 0.75rem; 
            font-weight: 500;
            text-transform: uppercase; 
            letter-spacing: 0.05em; 
            color: var(--ws-muted);
            margin-bottom: 0.25rem;
        }}
        
        /* KPI numbers: large + strong */
        .ws-kpi-value {{ 
            font-size: 2.25rem; 
            font-weight: 700; 
            color: var(--ws-midnight); 
            line-height: 1.1;
            margin: 0;
        }}
        
        /* Secondary text: muted grey */
        .ws-secondary {{ 
            color: var(--ws-muted); 
            font-size: 0.875rem;
            font-weight: 400;
        }}
        
        /* Micro labels */
        .ws-micro {{ 
            font-size: 0.75rem; 
            color: var(--ws-muted);
            font-weight: 400;
        }}
        
        /* Main container - more whitespace, less borders */
        .ws-main {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 2rem 2rem 3rem 2rem; 
        }}
        
        /* Subtle dividers instead of borders */
        .ws-divider {{ 
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, var(--ws-stone) 50%, transparent 100%);
            margin: 2rem 0;
        }}
        
        /* KPI Cards - flat, minimal shadows */
        .ws-kpi-card {{ 
            background: var(--ws-off-white);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: var(--ws-radius);
            padding: 1.5rem;
            text-align: center;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }}
        
        .ws-kpi-card:hover {{ 
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transform: translateY(-1px);
        }}
        
        /* Action Cards - disciplined color system */
        .ws-action-card {{ 
            background: var(--ws-off-white);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: var(--ws-radius);
            padding: 1.25rem;
            margin-bottom: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .ws-action-card.urgent {{ 
            border-left: 4px solid var(--ws-red);
            background: linear-gradient(135deg, #FEF2F2 0%, var(--ws-off-white) 100%);
        }}
        
        .ws-action-card.review {{ 
            border-left: 4px solid var(--ws-amber);
            background: linear-gradient(135deg, #FFFBEB 0%, var(--ws-off-white) 100%);
        }}
        
        .ws-action-card.opportunity {{ 
            border-left: 4px solid var(--ws-green);
            background: linear-gradient(135deg, #F0FDF4 0%, var(--ws-off-white) 100%);
        }}
        
        .ws-action-card:hover {{ 
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        /* Badges - disciplined color system */
        .ws-badge {{ 
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .ws-badge.risk {{ background: var(--ws-red); color: white; }}
        .ws-badge.review {{ background: var(--ws-amber); color: white; }}
        .ws-badge.opportunity {{ background: var(--ws-green); color: white; }}
        .ws-badge.neutral {{ background: var(--ws-muted); color: white; }}
        
        /* Buttons - clean, purposeful */
        .stButton > button {{ 
            border-radius: var(--ws-radius);
            border: 1px solid transparent;
            font-weight: 500;
            padding: 0.75rem 1.5rem;
            transition: all 0.2s ease;
            font-family: 'Inter', sans-serif;
        }}
        
        .stButton > button:hover {{ 
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .stButton > button[kind="primary"] {{ 
            background-color: var(--ws-primary);
            color: var(--ws-midnight);
            border-color: var(--ws-primary);
        }}
        
        /* Decision controls - prominent */
        .ws-decision-controls {{ 
            background: rgba(255,181,71,0.08);
            border-radius: var(--ws-radius);
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255,181,71,0.2);
        }}
        
        /* Sidebar - clean, flat */
        section[data-testid="stSidebar"] > div {{ 
            background-color: #FAFAFA;
            border-right: 1px solid rgba(0,0,0,0.06);
        }}
        
        .sidebar-nav {{ margin-bottom: 1rem; }}
        
        .nav-item {{ 
            font-size: 0.95rem; 
            padding: 0.75rem 1rem; 
            margin: 0.25rem 0; 
            border-radius: 6px; 
            color: var(--ws-midnight);
            font-weight: 500;
            transition: all 0.15s ease;
        }}
        
        .nav-active {{ 
            background: var(--ws-primary);
            color: var(--ws-midnight);
            font-weight: 600;
        }}
        
        .sidebar-section {{ 
            font-size: 0.7rem; 
            text-transform: uppercase; 
            letter-spacing: 0.05em; 
            color: var(--ws-muted); 
            margin: 1.5rem 0 0.5rem 0;
            font-weight: 600;
        }}
        
        .sidebar-context {{ 
            font-size: 0.8rem; 
            color: var(--ws-muted); 
            margin: 0.5rem 0; 
            line-height: 1.4;
        }}
        
        /* Remove Streamlit's default borders and boxes */
        .stDataFrame {{ border: none !important; }}
        .stDataFrame > div {{ border: none !important; }}
        section[data-testid="stSidebar"] .stExpander {{ 
            border: none !important; 
            background: transparent !important; 
            box-shadow: none !important; 
        }}
        
        /* System status indicators */
        .ws-status-indicator {{ 
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }}
        
        .ws-status-indicator.healthy {{ background: var(--ws-green); }}
        .ws-status-indicator.warning {{ background: var(--ws-amber); }}
        .ws-status-indicator.error {{ background: var(--ws-red); }}
        
        /* Audit trail */
        .ws-audit-summary {{ 
            background: rgba(0,0,0,0.02);
            border-radius: var(--ws-radius);
            padding: 1rem;
            margin: 1rem 0;
            font-size: 0.875rem;
        }}
        
        /* Model confidence context */
        .ws-model-status {{ 
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
        }}
        
        .ws-model-status .target {{ color: var(--ws-muted); }}
        .ws-model-status .current {{ font-weight: 600; }}
        .ws-model-status .action {{ color: var(--ws-amber); font-weight: 500; }}
        
        /* Tight spacing — control room feel, not document */
        .ws-main {{ padding-top: 0.4rem !important; padding-bottom: 1.5rem !important; }}
        .ws-page-title {{ margin-top: 0 !important; margin-bottom: 0.2rem !important; font-size: 1.6rem !important; }}
        [data-testid="stAppViewContainer"] {{ padding-top: 0.5rem !important; }}
        [data-testid="block-container"] {{ padding-top: 0.5rem !important; padding-bottom: 1rem !important; }}
        .ws-secondary {{ margin-bottom: 0.1rem !important; }}
        .ws-divider {{ margin: 0.6rem 0 !important; }}
        .ws-section-header {{ margin-top: 0.25rem !important; margin-bottom: 0.3rem !important; }}
        
        /* PART 2: Audit Status compact horizontal layout */
        .ws-audit-status-row {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin-bottom: 0.5rem; }}
        .ws-audit-kpi {{ display: flex; flex-direction: column; gap: 0.15rem; }}
        .ws-audit-kpi-label {{ font-size: 0.7rem; color: var(--ws-muted); text-transform: uppercase; }}
        .ws-audit-kpi-value {{ font-weight: 600; font-size: 0.9rem; }}
        
        /* Case Queue — operational cards (Decision Console) via st.container */
        [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-tier-red),
        [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-tier-amber),
        [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-tier-green) {{
            border-left-width: 4px !important;
            border-radius: var(--ws-radius) !important;
            margin-bottom: 0.5rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-tier-red) {{ border-left-color: #dc2626 !important; }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-tier-amber) {{ border-left-color: #d97706 !important; }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-tier-green) {{ border-left-color: #16a34a !important; }}
        .cc-tier-red, .cc-tier-amber, .cc-tier-green {{ display: none !important; }}
        .tier-segment {{ 
            display: inline-flex; border-radius: 6px; padding: 2px; 
            background: var(--ws-stone); gap: 2px;
        }}
        .tier-segment-btn {{ 
            padding: 0.4rem 0.75rem; border-radius: 4px; font-size: 0.85rem; font-weight: 500;
            border: none; cursor: pointer; background: transparent;
        }}
        .tier-segment-btn.active {{ background: white; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
        .queue-progress {{ font-size: 0.85rem; color: var(--ws-muted); margin-bottom: 0.75rem; }}
        
        /* Action Stack: left border + background on the row (stHorizontalBlock), not on children */
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-highrisk),
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-batch),
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-growth) {{
            gap: 0 !important;
            margin-bottom: 0.4rem !important;
            box-sizing: border-box !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-highrisk) {{
            border-left: 4px solid #dc2626 !important;
            background: #fafafa !important;
            border-radius: 6px !important;
            align-items: stretch !important;
            min-height: 2.75rem;
        }}
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-batch) {{
            border-left: 4px solid #d97706 !important;
            background: #fafafa !important;
            border-radius: 6px !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-growth) {{
            border-left: 4px solid #16a34a !important;
            background: #fafafa !important;
            border-radius: 6px !important;
        }}
        /* Right column: right-align the button container (stElementContainer) in its parent (stVerticalBlock) */
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-highrisk) > div:last-child,
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-batch) > div:last-child,
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-growth) > div:last-child {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            padding-right: 20px !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-highrisk) > div:last-child [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-batch) > div:last-child [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-growth) > div:last-child [data-testid="stVerticalBlock"] {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            width: 100% !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-highrisk) > div:last-child [data-testid="stElementContainer"],
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-batch) > div:last-child [data-testid="stElementContainer"],
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-growth) > div:last-child [data-testid="stElementContainer"] {{
            margin-left: auto !important;
        }}
        /* High-risk row: card's parent = 100% of row height */
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-highrisk) > div:first-child {{
            height: 100% !important;
            min-height: 100% !important;
            display: flex !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.ws-action-card-highrisk) > div:first-child > div {{
            height: 100% !important;
            min-height: 100% !important;
            width: 100% !important;
        }}
        .ws-action-card-highrisk {{
            min-height: 100% !important;
            height: 100% !important;
            box-sizing: border-box;
        }}
        
        /* Status strip — neutral when safe, prominent only when degraded */
        .ws-status-strip {{
            font-size: 0.75rem;
            color: var(--ws-muted);
            padding: calc(0.3rem + 5px) 0;
            margin-bottom: 0.4rem;
            line-height: 1.4;
        }}
        .ws-status-strip .sep {{ color: rgba(0,0,0,0.12); }}
        .ws-status-strip.degraded {{
            background: #FFFBEB;
            border: 1px solid rgba(217,119,6,0.2);
            border-radius: var(--ws-radius);
            color: #92400e;
            padding: 0.4rem 0.75rem;
            font-weight: 500;
        }}
        
        /* Severity-based button overrides */
        .btn-urgent .stButton > button {{
            background: #dc2626 !important; color: white !important;
            border-color: #dc2626 !important; font-weight: 600;
        }}
        .btn-urgent .stButton > button:hover {{
            background: #b91c1c !important; border-color: #b91c1c !important;
        }}
        .btn-amber .stButton > button {{
            background: transparent !important; color: #92400e !important;
            border: 1.5px solid #d97706 !important; font-weight: 600;
        }}
        .btn-amber .stButton > button:hover {{ background: #FFFBEB !important; }}
        .btn-growth .stButton > button {{
            background: transparent !important; color: #15803d !important;
            border: 1.5px solid #16a34a !important; font-weight: 600;
        }}
        .btn-growth .stButton > button:hover {{ background: #F0FDF4 !important; }}
        .btn-muted .stButton > button {{
            background: transparent !important; color: var(--ws-muted) !important;
            border: 1px solid rgba(0,0,0,0.15) !important; font-weight: 500;
            font-size: 0.8rem !important;
        }}
        .btn-muted .stButton > button:hover {{
            background: var(--ws-stone) !important;
        }}
        /* Impact row: grey background card */
        [data-testid="stHorizontalBlock"]:has(.ws-impact-card) {{
            background: #fafafa !important;
            border-radius: 6px !important;
            margin-bottom: 0.4rem !important;
            align-items: center !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.ws-impact-card) > div:first-child {{
            display: flex !important;
            align-items: center !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.ws-impact-card) > div:first-child [data-testid="stVerticalBlock"] {{
            display: flex !important;
            justify-content: center !important;
            gap: 0 !important;
        }}
        /* Impact row: right-align the Growth Engine CTA */
        [data-testid="stHorizontalBlock"]:has(.btn-muted) > div:last-child {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            padding-right: 20px !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.btn-muted) > div:last-child [data-testid="stVerticalBlock"] {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            width: 100% !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.btn-muted) > div:last-child [data-testid="stElementContainer"] {{
            margin-left: auto !important;
        }}
        
        /* Tighter vertical rhythm (overrides applied per-page via scoped rules above) */
        
        /* Compact KPI row — horizontal, under actions */
        .ws-kpi-compact {{
            display: flex !important;
            flex-direction: row !important;
            gap: 2rem;
            flex-wrap: wrap;
            align-items: flex-start;
            padding: 0.6rem 0;
        }}
        .ws-kpi-compact-item {{
            display: inline-flex !important;
            flex-direction: column;
            gap: 0.1rem;
            min-width: 80px;
        }}
        .ws-kpi-compact-item .val {{ font-size: 1.1rem; font-weight: 700; color: var(--ws-midnight); }}
        .ws-kpi-compact-item .lbl {{ font-size: 0.7rem; color: var(--ws-muted); text-transform: uppercase; }}
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


# ---------------------------------------------------------------------------
# Reusable UI Components for Product-Grade Polish
# ---------------------------------------------------------------------------

def render_kpi_card(label: str, value: str, delta: str = None, delta_type: str = "neutral"):
    """Render a polished KPI card with strong typography hierarchy."""
    delta_class_map = {
        "positive": "opportunity",
        "negative": "risk", 
        "neutral": "neutral"
    }
    delta_class = delta_class_map.get(delta_type, "neutral")
    delta_html = f'<div class="ws-micro ws-badge {delta_class}" style="margin-top: 0.5rem;">{delta}</div>' if delta else ""
    
    st.markdown(f"""
    <div class="ws-kpi-card">
        <div class="ws-kpi-value">{value}</div>
        <div class="ws-kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_action_card(title: str, subtitle: str, action_text: str, urgency: str = "normal", key: str = None):
    """Render an actionable alert card with disciplined color system."""
    urgency_map = {
        "urgent": "urgent",
        "review": "review", 
        "growth": "opportunity",
        "normal": "review"
    }
    card_class = f"ws-action-card {urgency_map.get(urgency, 'review')}"
    
    st.markdown(f"""
    <div class="{card_class}">
        <div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 1rem;">{title}</div>
        <div class="ws-secondary" style="margin-bottom: 0.75rem;">{subtitle}</div>
        <div class="ws-micro" style="font-weight: 500; color: var(--ws-primary);">{action_text} →</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Return True if clicked
    return st.button(f"Execute: {action_text}", key=key, use_container_width=True, type="primary")


def render_governance_badge(tier: str) -> str:
    """Render a governance tier badge with disciplined color system."""
    badge_map = {
        "green": ("Green", "opportunity"),
        "amber": ("Amber", "review"), 
        "red": ("Red", "risk")
    }
    
    if tier.lower() in badge_map:
        label, css_class = badge_map[tier.lower()]
        return f'<span class="ws-badge {css_class}">{label}</span>'
    return f'<span class="ws-badge neutral">{tier}</span>'


def render_significance_badge(is_significant: bool, sample_size: int = None) -> str:
    """Render a significance badge with explanation."""
    if is_significant:
        label = f"Significant (n={sample_size:,})" if sample_size else "Significant"
        return f'<span class="ws-badge opportunity">{label}</span>'
    else:
        label = f"Insufficient Data (n={sample_size:,})" if sample_size else "Insufficient Data"
        return f'<span class="ws-badge neutral">{label}</span>'


def show_toast(message: str, duration: int = 3):
    """Show a toast notification (simplified implementation)."""
    st.success(message)  # Using Streamlit's built-in for now
    

def render_empty_state(title: str, subtitle: str, icon: str = "📊"):
    """Render a polished empty state with proper typography."""
    st.markdown(f"""
    <div class="ws-empty-state">
        <div class="ws-empty-state-icon">{icon}</div>
        <div class="ws-subsection">{title}</div>
        <div class="ws-secondary">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def format_currency(amount: float, decimals: int = 2) -> str:
    """Format currency for display. Use with st.metric() or st.write() only—never in st.markdown() (avoids $ italics)."""
    try:
        x = float(amount)
    except (TypeError, ValueError):
        return "—"
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.0f}k"
    return f"${x:,.2f}".replace(".00", "")


def format_percent(value: float, decimals: int = 2) -> str:
    """Format percentage; max 2 decimals. Use for display only."""
    try:
        return f"{float(value):.{decimals}%}"
    except (TypeError, ValueError):
        return "—"


def format_percentage(value: float) -> str:
    """Format percentage with 2 decimal places."""
    return format_percent(value, 2)


def format_number(value: float) -> str:
    """Format number with appropriate precision."""
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.0f}k"
    else:
        return f"{value:.0f}"


def get_last_updated() -> str:
    """Get a formatted 'last updated' timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def get_model_version() -> str:
    """Get current model version with build info."""
    return "v1.2.3-prod"  # Placeholder for prototype


def get_compliance_info():
    """Get compliance and audit information for fintech deployment."""
    from datetime import datetime, timedelta
    import random
    
    # Mock realistic compliance data
    now = datetime.now()
    
    return {
        "model_version": "v1.2.3-prod",
        "model_build_date": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
        "last_audit": (now - timedelta(days=random.randint(15, 30))).strftime("%Y-%m-%d"),
        "next_audit_due": (now + timedelta(days=random.randint(30, 60))).strftime("%Y-%m-%d"),
        "decisions_logged_today": random.randint(45, 78),
        "override_rate_30d": round(random.uniform(8, 15), 1),
        "compliance_status": "COMPLIANT",
        "data_retention_days": 2555,  # 7 years
        "last_export": (now - timedelta(hours=random.randint(2, 8))).strftime("%H:%M"),
    }


def compute_priority_score(hypothesis: dict, metrics_df: pd.DataFrame = None) -> float:
    """Compute a composite priority score for ranking."""
    base_confidence = hypothesis.get("confidence", 0.0)
    
    # Add uplift weight if available
    uplift_weight = 0.0
    if metrics_df is not None and not metrics_df.empty:
        # Simplified uplift lookup
        uplift_weight = 0.1  # Placeholder
    
    # Risk penalty for high-risk cases
    risk_penalty = 0.0
    governance = hypothesis.get("governance", {})
    if governance.get("tier") == "red":
        risk_penalty = 0.2
    
    return base_confidence + uplift_weight - risk_penalty


# ---------------------------------------------------------------------------
# System Maturity Signals
# ---------------------------------------------------------------------------

def render_audit_summary():
    """Render audit trail summary for system maturity (legacy bullet list). Prefer render_audit_status() for Decision Console."""
    compliance = get_compliance_info()
    override_pct = compliance['override_rate_30d']
    override_color = "#d97706" if override_pct > 10 else "#16a34a"
    status = compliance['compliance_status']
    status_color = "#16a34a" if status == "COMPLIANT" else "#dc2626"
    st.markdown(f"""
    <div style="margin-top:0.5rem;">
        <div class="ws-subsection" style="margin: 0 0 0.75rem 0;">Audit & Compliance</div>
        <div style="background:#f1f5f9;border-radius:6px;padding:0.6rem 0.9rem;font-size:0.82rem;color:#334155;line-height:1.8;">
            <span class="ws-status-indicator healthy"></span><strong style="color:#16a34a;">100%</strong> decisions logged (retention: <strong>{compliance['data_retention_days']}</strong> days)<br>
            <span class="ws-status-indicator warning"></span>Override rate: <strong style="color:{override_color};">{override_pct}%</strong> (30d rolling)<br>
            <span class="ws-status-indicator healthy"></span>Status: <span style="background:{status_color};color:white;padding:0.1rem 0.5rem;border-radius:4px;font-weight:600;font-size:0.75rem;">{status}</span><br>
            <span class="ws-status-indicator healthy"></span>Last export: <strong>{compliance['last_export']}</strong> | Next audit: <strong>{compliance['next_audit_due']}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_audit_status():
    """Compact executive-grade Audit Status trust panel. <20% above-the-fold."""
    compliance = get_compliance_info()
    status = compliance["compliance_status"]
    badge_class = "opportunity" if status == "COMPLIANT" else "risk"
    override_pct = compliance["override_rate_30d"]  # already a number like 10.7
    st.markdown('<div class="ws-subsection" style="margin-bottom: 0.5rem;">Audit Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ws-audit-status-row">
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Decision Logging</span><span class="ws-audit-kpi-value">100%</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Retention</span><span class="ws-audit-kpi-value">{compliance["data_retention_days"]} days</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Override Rate (30d)</span><span class="ws-audit-kpi-value">{override_pct:.1f}%</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Compliance</span><span class="ws-badge {badge_class}">{status}</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Last Export</span><span class="ws-audit-kpi-value">{compliance["last_export"]}</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Next Audit</span><span class="ws-audit-kpi-value">{compliance["next_audit_due"]}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_client_snapshot(hypothesis: dict, features: pd.DataFrame = None, last_reviewed: str = None):
    """Client Snapshot summary at top of profile: Persona, AUA, Signal, Confidence, Governance, Projected Impact, Liquidity, Last Reviewed."""
    user_id = hypothesis.get("user_id", "")
    gov = hypothesis.get("governance", {})
    trace = hypothesis.get("traceability", {})
    sb = trace.get("spending_buffer", {})
    tp = trace.get("target_product", {})

    # AUA: from latest feature row or placeholder
    aua_val = "—"
    if features is not None and not features.empty and "user_id" in features.columns and "aua_current" in features.columns:
        uf = features[features["user_id"] == user_id]
        if not uf.empty:
            aua_val = format_currency(float(uf.sort_values("month").iloc[-1]["aua_current"]))

    persona = TIER_LABELS.get(hypothesis.get("persona_tier", ""), hypothesis.get("persona_tier", "—"))
    signal = SIGNAL_LABELS.get(hypothesis.get("signal", ""), hypothesis.get("signal", "—"))
    conf = hypothesis.get("confidence")
    conf_str = format_percent(conf, 2) if conf is not None else "—"
    gov_label = gov.get("label", gov.get("tier", "—"))
    gov_icon = GOV_TIER_ICONS.get(gov.get("tier", ""), "")
    projected = tp.get("suggested_amount") or tp.get("projected_yield") or "—"
    if isinstance(projected, (int, float)):
        projected = format_currency(projected)
    runway = sb.get("months_of_runway")
    liquidity = f"{runway:.1f} mo runway" if runway is not None else "—"
    if runway is not None and runway < 3:
        liquidity += " (low)"
    last_rev = last_reviewed or "—"

    st.markdown('<div class="ws-subsection" style="margin-bottom: 0.5rem;">Client Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ws-audit-status-row ws-client-snapshot">
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Persona</span><span class="ws-audit-kpi-value">{persona}</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">AUA</span><span class="ws-audit-kpi-value">{aua_val}</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Active Signal</span><span class="ws-audit-kpi-value">{signal}</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Confidence</span><span class="ws-audit-kpi-value">{conf_str}</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Governance</span><span class="ws-audit-kpi-value">{gov_icon} {gov_label}</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Projected Impact</span><span class="ws-audit-kpi-value">{projected}</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Liquidity</span><span class="ws-audit-kpi-value">{liquidity}</span></div>
        <div class="ws-audit-kpi"><span class="ws-audit-kpi-label">Last Reviewed</span><span class="ws-audit-kpi-value">{last_rev}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_governance_constraints():
    """Show explicit hard constraints for trust and compliance."""
    st.markdown("""
    <div style="margin-bottom:0.75rem;">
        <div class="ws-subsection" style="margin: 0 0 0.75rem 0;">Governance Thresholds</div>
        <div style="background:#f1f5f9;border-radius:6px;padding:0.6rem 0.9rem;font-size:0.82rem;color:#334155;line-height:1.8;">
            <strong style="color:#0f172a;">Auto-Escalation Rules:</strong><br>
            • Illiquid allocation <span style="color:#dc2626;font-weight:600;">&gt;20% AUA</span> → Manual review required<br>
            • Credit exposure <span style="color:#dc2626;font-weight:600;">&gt;5x monthly income</span> → Compliance review<br>
            • Model confidence <span style="color:#d97706;font-weight:600;">&lt;0.60</span> → Auto-approval blocked<br>
            • Product value <span style="color:#d97706;font-weight:600;">&gt;$50k</span> → Senior approval required
        </div>
        <div style="background:#f0fdf4;border-radius:6px;padding:0.6rem 0.9rem;font-size:0.82rem;color:#334155;line-height:1.8;margin-top:0.5rem;">
            <strong style="color:#0f172a;">Regulatory Compliance:</strong><br>
            • PIPEDA privacy impact assessed <span style="color:#16a34a;font-weight:600;">✓</span><br>
            • OSFI ML/AI guidelines compliant <span style="color:#16a34a;font-weight:600;">✓</span><br>
            • Suitability determination documented <span style="color:#16a34a;font-weight:600;">✓</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_model_confidence_context(persona: str, current: float, target: float = 0.75):
    """Show model confidence with operational context."""
    if current >= target:
        status_class = "healthy"
        action = "Within target"
        action_color = "#16a34a"
        bg = "#f0fdf4"
    elif current >= target * 0.8:
        status_class = "warning"
        action = "Monitoring — retraining scheduled"
        action_color = "#d97706"
        bg = "#fffbeb"
    else:
        status_class = "error"
        action = "Below threshold — retraining in progress"
        action_color = "#dc2626"
        bg = "#fef2f2"

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.6rem;background:{bg};border-radius:6px;padding:0.45rem 0.75rem;margin-bottom:0.35rem;font-size:0.82rem;">
        <span class="ws-status-indicator {status_class}"></span>
        <span style="font-weight:700;color:#0f172a;min-width:110px;">{persona}</span>
        <span style="color:#64748b;">Target: {target:.2f}</span>
        <span style="font-weight:700;color:#0f172a;">Current: {current:.2f}</span>
        <span style="color:{action_color};font-weight:600;">{action}</span>
    </div>
    """, unsafe_allow_html=True)


def get_system_timestamps():
    """Get formatted system timestamps for maturity signals."""
    from datetime import datetime, timedelta
    import random
    
    # Mock realistic timestamps for prototype
    now = datetime.now()
    last_updated = now - timedelta(minutes=random.randint(1, 5))
    last_retrain = now - timedelta(days=random.randint(5, 10))
    
    return {
        "last_updated": last_updated.strftime("%H:%M"),
        "last_retrain": f"{(now - last_retrain).days} days ago",
        "next_retrain": "3 days"
    }


def show_micro_feedback_toast(message: str, success: bool = True):
    """Show micro-feedback with animation."""
    toast_class = "opportunity" if success else "risk"
    
    # Use Streamlit's built-in for now, but with custom styling
    if success:
        st.success(f"✓ {message}")
    else:
        st.error(f"✗ {message}")
    
    # In production, this would trigger a JavaScript animation


def generate_decision_log_export():
    """Generate a mock decision log export for compliance."""
    import pandas as pd
    from datetime import datetime, timedelta
    import random
    
    # Mock decision log data for compliance export
    decisions = []
    personas = ["aspiring_affluent", "sticky_family_leader", "generation_nerd"]
    signals = ["leapfrog_ready", "liquidity_warning", "harvest_opportunity"]
    products = ["RRSP_LOAN", "SUMMIT_PE", "DIRECT_INDEX"]
    actions = ["approved", "rejected", "pending"]
    
    for i in range(50):  # 50 recent decisions
        timestamp = datetime.now() - timedelta(hours=random.randint(1, 168))  # Last week
        user_id = f"usr_{random.randint(10000, 99999)}"
        
        decisions.append({
            "decision_id": f"dec_{timestamp.strftime('%Y%m%d')}_{i:03d}",
            "timestamp": timestamp.isoformat(),
            "user_id": user_id,
            "curator_id": f"curator_{random.randint(1, 5)}",
            "persona_tier": random.choice(personas),
            "signal": random.choice(signals),
            "product_code": random.choice(products),
            "model_confidence": round(random.uniform(0.45, 0.95), 3),
            "governance_tier": random.choice(["green", "amber", "red"]),
            "decision": random.choice(actions),
            "override_reason": "Manual review" if random.random() < 0.15 else None,
            "model_version": "v1.2.3-prod",
            "macro_context": f"BoC:{random.uniform(3.5, 5.5):.2f}%,VIX:{random.randint(15, 30)}",
        })
    
    df = pd.DataFrame(decisions)
    return df


def render_compliance_export_section():
    """Render compliance export controls."""
    st.markdown('<div class="ws-section-header">Compliance & Audit</div>', unsafe_allow_html=True)
    
    compliance = get_compliance_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="ws-subsection">Decision Log Export</div>', unsafe_allow_html=True)
        
        export_col1, export_col2 = st.columns([2, 1])
        with export_col1:
            date_range = st.selectbox(
                "Export Period",
                ["Last 7 days", "Last 30 days", "Last 90 days", "Custom range"],
                key="export_period"
            )
        with export_col2:
            if st.button("📋 Export CSV", key="export_decisions"):
                # Generate mock export
                df = generate_decision_log_export()
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download",
                    data=csv,
                    file_name=f"pulse_decisions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
                show_micro_feedback_toast("Decision log exported for compliance review", success=True)
        
        st.markdown(f"""
        <div class="ws-secondary" style="margin-top: 0.5rem;">
        Last export: {compliance['last_export']} | {compliance['decisions_logged_today']} decisions today
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="ws-subsection">Override Audit Trail</div>', unsafe_allow_html=True)
        
        if st.button("📊 View Override Report", key="override_report"):
            show_micro_feedback_toast("Override audit report generated", success=True)
            
        st.markdown(f"""
        <div class="ws-secondary">
        Override rate (30d): {compliance['override_rate_30d']}%<br>
        Threshold: <15% (compliant)<br>
        Last review: {compliance['last_audit']}
        </div>
        """, unsafe_allow_html=True)


def render_model_governance_panel():
    """Render model version and governance information."""
    compliance = get_compliance_info()
    
    st.markdown(f"""
    <div class="ws-audit-summary">
        <div class="ws-subsection" style="margin: 0 0 0.75rem 0;">Model Governance</div>
        <div class="ws-secondary">
            <strong>Production Model:</strong> {compliance['model_version']}<br>
            <strong>Build Date:</strong> {compliance['model_build_date']}<br>
            <strong>Validation Status:</strong> Approved for production use<br>
            <strong>Performance Monitoring:</strong> Active<br>
            <strong>Drift Detection:</strong> Enabled (±5% threshold)<br>
            <strong>Retraining Schedule:</strong> Monthly or on drift detection<br><br>
            
            <strong>Approval Chain:</strong><br>
            • Model Risk: ✓ Approved ({compliance['model_build_date']})<br>
            • Compliance: ✓ Approved ({compliance['last_audit']})<br>
            • IT Security: ✓ Approved ({compliance['model_build_date']})<br>
            • Business Owner: ✓ Approved ({compliance['model_build_date']})
        </div>
    </div>
    """, unsafe_allow_html=True)


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
# Model artifact loading (for Growth Engine and sidebar status)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts() -> dict:
    """Load per-persona joblib artifacts. Returns {persona: artifact_dict}. Cached so sidebar stays fast on page switch."""
    artifacts = {}
    for persona in PERSONAS:
        path = MODEL_DIR / f"model_{persona}.joblib"
        if path.exists():
            artifacts[persona] = joblib.load(path)
    return artifacts


def get_system_status_labels():
    """Macro and models status for sidebar. Returns (macro_label, models_label)."""
    macro = st.session_state.get("macro")
    if macro is None:
        macro = get_default_macro()
    rate_label = "High" if macro.rates_high else "Normal"
    vol_label = "Elevated" if macro.market_volatile else "Normal"
    macro_label = f"Macro: {rate_label} (BoC {macro.boc_prime_rate:.2f}%, VIX {macro.vix:.0f})"

    artifacts = load_model_artifacts()
    if not artifacts:
        models_label = "Models: —"
    else:
        precision_target = 0.80
        below = sum(1 for a in artifacts.values() if (a.get("metrics", {}).get("precision", 0) or 0) < precision_target)
        if below == 0:
            models_label = "Models: Stable"
        else:
            models_label = f"Models: Mixed (⚠ {below} below threshold)"
    return macro_label, models_label


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


# ---------------------------------------------------------------------------
# Shared command-console sidebar (Command Layer → Execution → Intelligence)
# ---------------------------------------------------------------------------
NAV_PAGES = [
    ("control", "Control Center", "app.py"),
    ("decision", "Decision Console", "pages/1_decision_console.py"),
    ("growth", "Growth Engine", "pages/2_growth_engine.py"),
    ("compliance", "Compliance & Audit", "pages/3_compliance.py"),
]


def render_pulse_sidebar(current_page: str):
    """
    Executive command console sidebar: Primary nav, System Status, Configure, Help.
    Clean hierarchy with no redundant headers or bordered cards.
    """
    if "macro" not in st.session_state:
        st.session_state.macro = get_default_macro()

    sb = st.sidebar
    macro = st.session_state.macro

    # ----- Primary Navigation (top, high emphasis) -----
    sb.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
    for page_id, label, path in NAV_PAGES:
        if page_id == current_page:
            sb.markdown(f'<div class="nav-item nav-active">{label}</div>', unsafe_allow_html=True)
        else:
            try:
                sb.page_link(path, label=label)
            except Exception:
                if sb.button(label, key=f"nav_{page_id}"):
                    st.switch_page(path)
    sb.markdown("</div>", unsafe_allow_html=True)
    sb.markdown("---")

    # ----- System Status (secondary, compact) -----
    sb.markdown('<p class="sidebar-section">System Status</p>', unsafe_allow_html=True)
    
    # Macro regime
    regime = "Normal" if 3.0 <= macro.boc_prime_rate <= 6.0 and macro.vix <= 25 else "Volatile"
    sb.markdown(f'<div class="sidebar-context">Macro: {regime} (BoC {macro.boc_prime_rate:.2f}% • VIX {macro.vix})</div>', 
                unsafe_allow_html=True)
    
    # Model health
    try:
        artifacts = load_model_artifacts()
        precision_issues = sum(1 for p in PERSONAS if artifacts.get(p, {}).get("precision", 1.0) < 0.75)
        if precision_issues == 0:
            health_status = "Precision Stable"
        else:
            health_status = f"Mixed (⚠ {precision_issues} below threshold)"
    except:
        health_status = "Unknown"
    
    sb.markdown(f'<div class="sidebar-context">Models: {health_status}</div>', unsafe_allow_html=True)
    
    # Compliance status
    compliance = get_compliance_info()
    status_indicator = "healthy" if compliance['compliance_status'] == "COMPLIANT" else "warning"
    sb.markdown(f'<div class="sidebar-context"><span class="ws-status-indicator {status_indicator}"></span>Compliance: {compliance['compliance_status']}</div>', 
                unsafe_allow_html=True)
    sb.markdown("---")

    # ----- Configure (collapsible, minimal) -----
    sb.markdown('<p class="sidebar-section">Configure</p>', unsafe_allow_html=True)
    
    with sb.expander("Filters", expanded=False):
        tier_options = [k for k in TIER_LABELS if k != "not_eligible"]
        tier_filter = st.multiselect(
            "Persona Tier",
            options=tier_options,
            default=tier_options,
            key="pulse_tier_filter",
            format_func=lambda x: TIER_LABELS[x].split("(")[0].strip(),
        )
        confidence_min = st.slider(
            "Min Confidence",
            0.0, 1.0, 0.5, 0.05,
            key="pulse_confidence_min",
        )

    # Static scenario panels (no interactive sliders — executive summary only)
    with sb.expander("Scenario", expanded=False):
        st.caption("BoC Rate Scenarios")
        boc_data = [
            ("Current (Baseline)", f"{macro.boc_prime_rate:.2f}%", "—", "0%", "—"),
            ("+1% Rate Increase", f"{macro.boc_prime_rate + 1:.2f}%", "-2 to -5%", "~8%", "-3%"),
            ("-1% Rate Decrease", f"{max(2, macro.boc_prime_rate - 1):.2f}%", "+1 to +3%", "~2%", "+2%"),
        ]
        df_boc = pd.DataFrame(boc_data, columns=["Scenario", "Rate", "Confidence impact", "Pathway suppression %", "Projected AUA change"])
        st.dataframe(df_boc, use_container_width=True, hide_index=True)
        st.caption("VIX Scenarios")
        vix_data = [
            ("Current (Baseline)", f"{macro.vix:.0f}", "—", "—", "—"),
            ("High Volatility (VIX +10)", f"{min(50, macro.vix + 10):.0f}", "Suppress", "Higher", "Growth deprioritized"),
            ("Low Volatility (VIX -10)", f"{max(10, macro.vix - 10):.0f}", "Allow", "Lower", "Growth prioritized"),
        ]
        df_vix = pd.DataFrame(vix_data, columns=["Scenario", "VIX", "Private market", "Risk suppression", "Growth"])
        st.dataframe(df_vix, use_container_width=True, hide_index=True)

    sb.markdown("---")

    # ----- Help (low emphasis) -----
    sb.markdown('<p class="sidebar-section">Help</p>', unsafe_allow_html=True)
    
    if sb.button("About", key="about_pulse"):
        with sb.expander("About Pulse", expanded=True):
            sb.markdown("**AI Growth Control Panel**")
            sb.caption("Monitors client personas for upgrade, risk, and opportunity signals.")
            
    if sb.button("Tour", key="start_tour"):
        from ui.onboarding import start_tour
        start_tour()
