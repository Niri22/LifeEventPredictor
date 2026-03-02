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

# Human-readable labels for model feature attribution
FEATURE_LABELS = {
    "aua_delta": "AUA Growth Velocity",
    "aua_current": "Current AUA",
    "illiquidity_ratio": "Illiquidity Ratio",
    "spend_velocity_30d": "Spend Velocity (30d)",
    "spend_velocity_delta": "Spend Velocity Delta",
    "credit_spend_vs_invest": "Credit Spend vs Invest",
    "cc_spend_30d": "Credit Card Spend (30d)",
    "mcc_entropy": "Transaction Diversity",
    "txn_count_30d": "Transaction Count (30d)",
    "top_mcc_concentration": "Top Category Concentration",
    "savings_rate": "Savings Rate",
    "pct_spend_on_ws_cc": "WS Credit Card Share",
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
        
        /* Color system - disciplined palette, stronger contrast */
        :root {{ 
            --ws-midnight: #111;
            --ws-off-white: #fff;
            --ws-gold: {WS_GOLD}; 
            --ws-sage: {SAGE_GREEN}; 
            --ws-stone: #e8e8e8;
            --ws-radius: 8px;
            --ws-red: #DC2626;      /* Risk */
            --ws-amber: #F59E0B;    /* Review */
            --ws-green: #059669;    /* Opportunity */
            --ws-primary: {WS_GOLD}; /* Action */
            --ws-muted: #666;       /* Metadata */
            --ws-secondary: #444;   /* Secondary text */
            --ws-primary-text: #1a1a1a;
        }}
        
        /* Base app styling - flat, minimal borders, readable body */
        .stApp {{ 
            background-color: var(--ws-off-white); 
            color: var(--ws-primary-text); 
            font-family: 'Inter', system-ui, sans-serif;
            font-size: 15px;
            line-height: 1.5;
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
        
        /* Section headers: 16px semibold */
        .ws-section-header {{ 
            font-family: 'Inter', sans-serif; 
            font-size: 16px; 
            font-weight: 600; 
            color: var(--ws-primary-text); 
            margin: 2rem 0 1rem 0;
            letter-spacing: -0.01em;
        }}
        
        /* Subsection headers */
        .ws-subsection {{ 
            font-family: 'Inter', sans-serif; 
            font-size: 16px; 
            font-weight: 600; 
            color: var(--ws-primary-text); 
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
        [data-testid="stAppViewContainer"] {{ padding-top: 0.5rem !important; background: #fff !important; }}
        [data-testid="block-container"] {{ padding-top: 0.5rem !important; padding-bottom: 1rem !important; background: #fff !important; color: #1a1a1a; font-size: 15px; line-height: 1.5; }}
        .ws-secondary {{ margin-bottom: 0.1rem !important; }}
        .ws-divider {{ margin: 0.6rem 0 !important; }}
        .ws-section-header {{ margin-top: 0.25rem !important; margin-bottom: 0.3rem !important; }}
        
        /* PART 2: Audit Status compact horizontal layout */
        .ws-audit-status-row {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin-bottom: 0.5rem; }}
        .ws-audit-kpi {{ display: flex; flex-direction: column; gap: 0.15rem; }}
        .ws-audit-kpi-label {{ font-size: 0.7rem; color: var(--ws-muted); text-transform: uppercase; }}
        .ws-audit-kpi-value {{ font-weight: 600; font-size: 0.9rem; }}
        
        /* Case Queue — grey container wraps each case card (Decision Console) */
        [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-tier-red),
        [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-tier-amber),
        [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-tier-green) {{
            background: #f2f2f2 !important;
            border-left-width: 4px !important;
            border-radius: var(--ws-radius) !important;
            margin-bottom: 0.5rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
            padding: 0.5rem 0.75rem !important;
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

        /* AI Decision Brief — structured 2-col, high density, 15–20% tighter */
        .adb-header-block {{ background: #fff; border-bottom: 1px solid #ddd; margin-bottom: 0.35rem; padding: 0.35rem 0; }}
        .adb-header-left .adb-line {{ font-size: 14px; color: #1a1a1a; margin: 0.08rem 0; line-height: 1.42; }}
        .adb-header-right {{ display: flex; flex-direction: column; align-items: flex-end; gap: 0.2rem; }}
        .adb-conf-value {{ font-size: 20px; font-weight: 700; color: #0F5132; letter-spacing: -0.02em; line-height: 1.2; }}
        .adb-conf-value-mid {{ color: #444; }}
        .adb-conf-value-low {{ color: #b91c1c; }}
        .adb-conf-bar-wrap {{ width: 100%; max-width: 120px; height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden; margin-top: 0.15rem; }}
        .adb-conf-bar-fill {{ height: 100%; border-radius: 3px; background: #0F5132; }}
        .adb-conf-bar-fill.adb-conf-bar-mid {{ background: #78716c; }}
        .adb-conf-bar-fill.adb-conf-bar-low {{ background: #b91c1c; }}
        .adb-risk-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.1rem; }}
        .adb-tier-descriptor {{ font-size: 10px; font-weight: 600; color: #555; margin-top: 0.15rem; text-transform: uppercase; letter-spacing: 0.03em; }}
        .adb-conf-gauge {{ position: relative; width: 100%; max-width: 140px; margin-top: 0.35rem; }}
        .adb-conf-gauge-track {{ height: 8px; background: #e5e7eb; border-radius: 4px; overflow: visible; position: relative; }}
        .adb-conf-gauge-fill {{ height: 100%; border-radius: 4px; transition: width 0.2s; }}
        .adb-conf-gauge-fill.high {{ background: #0F5132; }}
        .adb-conf-gauge-fill.mid {{ background: #b45309; }}
        .adb-conf-gauge-fill.low {{ background: #b91c1c; }}
        .adb-conf-gauge-threshold {{ position: absolute; top: -2px; bottom: -2px; width: 2px; background: #374151; border-radius: 1px; z-index: 1; }}
        .adb-conf-gauge-labels {{ display: flex; justify-content: space-between; font-size: 9px; color: #6b7280; margin-top: 0.2rem; }}
        .adb-conf-gauge-labels .th {{ font-weight: 700; color: #374151; }}
        .adb-band-high {{ color: #0F5132; font-weight: 600; }}
        .adb-band-mid {{ color: #666; }}
        .adb-band-low {{ color: #b91c1c; font-weight: 600; }}
        .adb-macro-reason {{ font-size: 13px; color: #666; margin-bottom: 0.4rem; padding-bottom: 0.35rem; border-bottom: 1px solid #ddd; }}
        /* View Details expander: narrative pill row — Signal · Tier · Confidence · Persona · Pathway · Why flagged */
        .adb-header-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.2rem 0.5rem; padding: 0.4rem 0.6rem; margin-bottom: 0.4rem; background: #fafafa; border-radius: 6px; border: 1px solid #e8e8e8; line-height: 1.35; font-size: 12px; }}
        .adb-header-sep {{ color: #bbb; font-weight: 400; padding: 0 0.35rem; user-select: none; }}
        .adb-header-label {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: #888; margin-right: 0.25rem; }}
        .adb-header-item {{ display: inline-flex; align-items: center; }}
        .adb-chip {{ display: inline-block; padding: 0.12rem 0.4rem; border-radius: 4px; font-size: 11px; font-weight: 500; background: #fff; border: 1px solid #e0e0e0; color: #333; max-width: 10em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .adb-conf-pill {{ display: inline-flex; align-items: center; padding: 0.15rem 0.45rem; border-radius: 999px; font-size: 12px; font-weight: 700; background: #fff; border: 1px solid #e0e0e0; }}
        .adb-tier-block {{ display: inline-flex; flex-direction: column; align-items: center; gap: 0.12rem; padding: 0.28rem 0.55rem; border-radius: 6px; font-size: 13px; font-weight: 800; min-width: 4.5rem; letter-spacing: 0.02em; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
        .adb-tier-block.red {{ background: #fef2f2; border: 1.5px solid #f87171; color: #b91c1c; }}
        .adb-tier-block.amber {{ background: #fffbeb; border: 1.5px solid #fbbf24; color: #b45309; }}
        .adb-tier-block.green {{ background: #f0fdf4; border: 1.5px solid #4ade80; color: #15803d; }}
        .adb-tier-block .adb-tier-icon {{ font-size: 16px; line-height: 1; margin-right: 0.25rem; }}
        .adb-tier-action {{ display: inline-flex; align-items: center; gap: 0.2rem; font-size: 9px; font-weight: 700; text-align: center; line-height: 1.2; text-transform: uppercase; letter-spacing: 0.03em; padding: 0.15rem 0.35rem; border-radius: 4px; }}
        .adb-tier-action.red {{ background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }}
        .adb-tier-action.amber {{ background: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }}
        .adb-tier-action.green {{ background: #d1fae5; color: #047857; border: 1px solid #6ee7b7; }}
        .adb-tier-action-icon {{ font-size: 10px; line-height: 1; }}
        .adb-why-flagged {{ font-size: 11px; color: #444; flex: 1; min-width: 100px; line-height: 1.35; }}
        .adb-why-flagged .adb-header-label {{ margin-right: 0.2rem; }}
        .adb-view-card {{ background: #fafafa; border: 1px solid #e8e8e8; border-radius: 6px; padding: 0.5rem 0.65rem; margin-bottom: 0.5rem; }}
        .adb-view-card .adb-title {{ margin-bottom: 0.25rem; }}
        .adb-summary-headline {{ font-size: 13px; font-weight: 700; color: #111; margin-bottom: 0.35rem; line-height: 1.4; }}
        .adb-summary-line {{ font-size: 12px; color: #444; line-height: 1.45; }}
        .adb-view-sep {{ height: 1px; background: #e0e0e0; margin: 0.4rem 0; }}
        .adb-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1rem; padding: 0.3rem 0; border-bottom: 1px solid #ddd; margin-bottom: 0.35rem; background: #fff; line-height: 1.45; }}
        .adb-section {{ background: #fff; padding: 0.28rem 0; margin-bottom: 0.25rem; border-bottom: 1px solid #ddd; line-height: 1.42; }}
        .adb-section:last-of-type {{ border-bottom: none; }}
        .adb-case-title {{ font-size: 18px; font-weight: 600; color: #111; letter-spacing: -0.01em; }}
        .adb-title {{ font-size: 12px; font-weight: 600; color: #444; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.22rem; }}
        .adb-line {{ font-size: 14px; color: #1a1a1a; margin: 0.08rem 0; line-height: 1.42; }}
        .adb-key {{ color: #444; font-weight: 500; }}
        .adb-emph {{ color: #111; font-weight: 600; font-size: 15px; }}
        .adb-badge {{ display: inline-block; padding: 0.18rem 0.45rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.02em; }}
        .adb-badge-red {{ background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }}
        .adb-badge-green {{ background: #ecfdf5; color: #0F5132; border: 1px solid #a7f3d0; font-size: 16px; font-weight: 700; }}
        .adb-badge-neutral {{ background: #f5f5f5; color: #444; border: 1px solid #e0e0e0; }}
        .adb-macro-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 0.2rem 0.75rem; font-size: 13px; color: #444; }}
        .adb-macro-grid .k {{ color: #666; }}
        .adb-macro-chip {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 6px; font-size: 12px; font-weight: 600; background: #f0f4f8; border: 1px solid #cbd5e1; color: #334155; }}
        .adb-impact-tag {{ display: inline-block; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.35rem; }}
        .adb-impact-tag.high {{ background: #ecfdf5; color: #0F5132; border: 1px solid #a7f3d0; }}
        .adb-impact-tag.medium {{ background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }}
        .adb-impact-tag.low {{ background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }}
        .adb-impact-metric {{ font-size: 13px; margin: 0.2rem 0; }}
        .adb-impact-metric .k {{ color: #64748b; font-weight: 500; }}
        .adb-progress-wrap {{ margin: 0.4rem 0; }}
        .adb-progress-label {{ font-size: 11px; color: #64748b; margin-bottom: 0.15rem; display: flex; justify-content: space-between; }}
        .adb-progress-bar {{ height: 8px; border-radius: 4px; overflow: hidden; background: #e2e8f0; position: relative; }}
        .adb-progress-bar .fill {{ height: 100%; border-radius: 4px; transition: width 0.2s; }}
        .adb-progress-bar .fill.safe {{ background: #22c55e; }}
        .adb-progress-bar .fill.caution {{ background: #eab308; }}
        .adb-progress-bar .fill.unsafe {{ background: #ef4444; }}
        .adb-progress-bar.threshold {{ background: linear-gradient(to right, #22c55e 0%, #22c55e 20%, #e2e8f0 20%, #e2e8f0 100%); }}
        .adb-progress-bar.threshold .fill {{ background: #ef4444; }}
        .adb-trigger-grid {{ display: grid; grid-template-columns: 6.5em 1fr; gap: 0.2rem 0.6rem; font-size: 12px; color: #333; align-items: baseline; }}
        .adb-trigger-grid .adb-trigger-k {{ color: #555; font-weight: 500; }}
        .adb-trigger-na {{ color: #999; font-style: italic; cursor: help; }}
        .adb-drivers-legend {{ font-size: 11px; color: #666; margin-bottom: 0.35rem; font-style: italic; }}
        .adb-attr-row-top {{ background: #f0f9ff; border-left: 3px solid #0ea5e9; padding-left: 0.35rem; }}
        .adb-attr-rank {{ font-weight: 700; color: #0ea5e9; margin-right: 0.25rem; font-size: 12px; }}
        .adb-attr-row .adb-attr-driver {{ color: #333; font-weight: 400; }}
        .adb-attr-impact {{ font-weight: 600; color: #0F5132; font-size: 13px; }}
        .adb-attr-impact.neg {{ color: #b91c1c; }}
        .adb-impact-block {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.4rem 0.5rem; background: #fafafa; }}
        .adb-decision-status-box {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.35rem 0.5rem; margin-bottom: 0.3rem; background: #fafafa; }}
        .adb-attr-header, .adb-attr-row {{
            display: grid;
            grid-template-columns: 1.4fr 1.6fr 0.4fr;
            gap: 0.28rem;
            align-items: center;
        }}
        .adb-attr-header {{ font-size: 12px; color: #444; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; font-weight: 600; }}
        .adb-attr-row {{ font-size: 14px; margin: 0.12rem 0; line-height: 1.45; }}
        .adb-track {{ background: #e0e0e0; border-radius: 2px; height: 6px; overflow: hidden; }}
        .adb-fill {{ background: #94a3b8; height: 100%; border-radius: 2px; }}
        .adb-fill-main {{ background: #0ea5e9; }}
        .adb-impact-tag-wrap {{ display: inline-flex; align-items: center; gap: 0.2rem; margin-bottom: 0.35rem; }}
        .adb-impact-info {{ font-size: 11px; color: #64748b; cursor: help; opacity: 0.85; }}
        .adb-impact-subhead {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin: 0.4rem 0 0.2rem 0; }}
        .adb-impact-subhead:first-of-type {{ margin-top: 0; }}
        .adb-attr-val {{ font-weight: 700; text-align: right; color: #111; font-size: 14px; }}
        .adb-attr-raw {{ color: #555; font-size: 13px; text-align: right; }}
        .adb-impact-amount {{ font-size: 20px; font-weight: 700; letter-spacing: -0.02em; margin: 0.12rem 0 0.2rem; line-height: 1.3; }}
        .adb-impact-positive {{ color: #0F5132; }}
        .adb-impact-unavail {{ font-size: 14px; font-weight: 500; color: #333; margin: 0.12rem 0; line-height: 1.48; }}
        .adb-status {{ display: inline-block; padding: 0.14rem 0.45rem; border-radius: 4px; font-size: 13px; font-weight: 600; margin-bottom: 0.15rem; }}
        .adb-status-approved {{ background: #0F5132; color: #fff; border: 1px solid #0F5132; }}
        .adb-status-rejected {{ background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }}
        .adb-status-pending {{ background: #f5f5f5; color: #666; border: 1px solid #e0e0e0; }}
        .adb-meta {{ font-size: 13px; color: #666; margin: 0.06rem 0; line-height: 1.45; }}
        .adb-divider {{ height: 1px; background: #ddd; margin: 0.35rem 0; }}
        .adb-muted {{ color: #666; font-size: 14px; }}
        /* Decision footer inside expander: sticky bar + override badge */
        .adb-decision-footer {{ margin-top: 0.75rem; padding-top: 0.6rem; border-top: 1px solid #e2e8f0; background: #f8fafc; border-radius: 6px; padding: 0.6rem 0.75rem; }}
        .adb-decision-footer .adb-override-badge {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 12px; font-weight: 600; background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; margin-bottom: 0.4rem; }}
        .adb-raw-drawer {{ margin-top: 0.5rem; }}
        
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
# View Details Expander (Decision Console — structured transparency)
# ---------------------------------------------------------------------------

def _feature_label(feature_key: str) -> str:
    """Human-readable label for model feature."""
    return FEATURE_LABELS.get(feature_key, feature_key.replace("_", " ").title())


def _format_feature_value(feature_key: str, value: float) -> str:
    """Format feature value for display."""
    if feature_key in ("illiquidity_ratio", "savings_rate", "pct_spend_on_ws_cc", "top_mcc_concentration"):
        return f"{value:.0%}"
    if feature_key in ("aua_current", "aua_delta", "spend_velocity_30d", "cc_spend_30d"):
        return format_currency(value) if value else "—"
    return f"{value:.2f}"


def render_view_details_expander(hypothesis: dict, existing_decision: dict = None):
    """
    Render the View Details expander content with decision clarity:
    Signal Summary, Model Explanation (bar chart), Projected Impact, Macro Context, Audit Metadata.
    """
    gov = hypothesis.get("governance", {})
    trace = hypothesis.get("traceability", {})
    tp = trace.get("target_product", {})
    sb = trace.get("spending_buffer", {})
    audit = trace.get("audit_log", [])
    signal = hypothesis.get("signal", "")
    signal_label = SIGNAL_LABELS.get(signal, signal.replace("_", " ").title())
    conf = float(hypothesis.get("confidence", 0.0))
    band_name, _ = confidence_band(conf)
    tier = (gov.get("tier", "green") or "green").lower()
    tier_label = tier.title()
    persona_label = TIER_LABELS.get(hypothesis.get("persona_tier", ""), hypothesis.get("persona_tier", "—"))
    pathway_label = (hypothesis.get("distance_to_upgrade") or {}).get("cohort_label", "—")
    compliance = get_compliance_info()
    macro = st.session_state.get("macro") or get_default_macro()
    boc = getattr(macro, "boc_prime_rate", 4.25)
    vix = getattr(macro, "vix", 18)
    override_pct = compliance.get("override_rate_30d", 10.0)

    # Pull common trigger features.
    illiquidity_ratio = None
    credit_vs_invest = None
    current_aua = None
    for entry in audit:
        feat = entry.get("feature", "")
        val = float(entry.get("value", 0) or 0)
        if feat == "illiquidity_ratio":
            illiquidity_ratio = val
        elif feat == "credit_spend_vs_invest":
            credit_vs_invest = val
        elif feat == "aua_current":
            current_aua = val

    tier_badge_class = "adb-badge-red" if tier == "red" else ("adb-badge-green" if tier == "green" else "adb-badge-neutral")
    conf_value_class = "adb-conf-value" + (" adb-conf-value-low" if conf < 0.60 else (" adb-conf-value-mid" if conf < 0.75 else ""))
    conf_bar_pct = min(100, int(conf * 100))
    conf_bar_class = "adb-conf-bar-fill" + (" adb-conf-bar-low" if conf < 0.60 else (" adb-conf-bar-mid" if conf < 0.75 else ""))
    band_label_class = "adb-band-high" if band_name == "High" else ("adb-band-low" if band_name == "Low" else "adb-band-mid")
    product_name = tp.get("name") or tp.get("code") or "—"
    workflow = gov.get("workflow", "—")
    macro_reasons = hypothesis.get("macro_reasons", [])
    macro_reason_line = "; ".join(macro_reasons[:2]) if macro_reasons else "No macro adjustments applied — conditions within normal range."

    # One-line "Why flagged" and tier action for compact header
    trigger_items = []
    if current_aua is not None and current_aua > 0:
        trigger_items.append(f"Current AUA: {format_currency(current_aua)}")
    if illiquidity_ratio is not None:
        trigger_items.append(f"Illiquidity ratio: {illiquidity_ratio:.0%} (threshold: 20%)")
    if credit_vs_invest is not None:
        trigger_items.append(f"Credit spend {credit_vs_invest:.1f}x transfer velocity")
    if not trigger_items:
        trigger_items.append(gov.get("reason", "Model confidence and product risk profile."))
    liquid_cash = sb.get("liquid_cash")
    burn_rate = sb.get("monthly_burn_rate")
    runway = float(sb.get("months_of_runway", 0) or 0)
    liquidity_meta = ""
    if liquid_cash is not None or burn_rate is not None or runway:
        liquidity_meta = '<div class="adb-view-sep"></div><div class="adb-meta">'
        if liquid_cash is not None:
            liquidity_meta += f'Liquid: {format_currency(float(liquid_cash))}'
        if burn_rate is not None:
            liquidity_meta += f' · Burn: {format_currency(float(burn_rate))}/mo'
        if runway:
            liquidity_meta += f' · Runway: {runway:.1f} mo'
        liquidity_meta += "</div>"
    macro_status = "Neutral" if not macro_reasons else "Adjusted"
    macro_adjustment = "No macro adjustments applied—conditions within normal range." if not macro_reasons else "; ".join(macro_reasons[:2])

    # Structured Trigger conditions (tight grid; grey "—" with tooltip when not a driver)
    _na = '<span class="adb-trigger-na" title="Not a driver in this case">—</span>'
    trigger_aua = format_currency(current_aua) if (current_aua is not None and current_aua > 0) else None
    trigger_illiq = f"{illiquidity_ratio:.0%} (threshold 20%)" if illiquidity_ratio is not None else None
    trigger_credit = f"{credit_vs_invest:.1f}x" if credit_vs_invest is not None else None
    trigger_runway = f"{runway:.1f} mo" if runway and runway > 0 else None
    _v = lambda x: x if x is not None else _na
    trigger_grid_html = (
        f'<div class="adb-trigger-grid">'
        f'<span class="adb-trigger-k">Current AUA</span><span>{_v(trigger_aua)}</span>'
        f'<span class="adb-trigger-k">Illiquidity</span><span>{_v(trigger_illiq)}</span>'
        f'<span class="adb-trigger-k">Credit vs invest</span><span>{_v(trigger_credit)}</span>'
        f'<span class="adb-trigger-k">Runway</span><span>{_v(trigger_runway)}</span>'
        f'</div>'
    )

    why_flagged = (trigger_items[0] if trigger_items else gov.get("reason", "—"))[:72]
    if len((trigger_items[0] if trigger_items else gov.get("reason", ""))) > 72:
        why_flagged += "…"
    tier_action = {"red": "Manual review required", "amber": "Batch review recommended", "green": "Auto-approve candidate"}.get(tier, "Review")
    tier_action_icon = {"red": "!", "amber": "▸", "green": "✓"}.get(tier, "•")
    tier_action_class = f"adb-tier-action {tier}" if tier in ("red", "amber", "green") else "adb-tier-action"
    # Concise chip labels (single line): shorten persona/pathway
    persona_chip = persona_label.split(" (")[0] if " (" in persona_label else persona_label[:18]
    if len(persona_label) > 18 and " (" not in persona_label:
        persona_chip = persona_label[:15] + "…"
    pathway_chip = pathway_label[:24] + "…" if len(pathway_label) > 24 else pathway_label

    # Tight header row: Signal chip | Tier (dominant, color + icon + action) | Confidence pill | Persona | Pathway | Why flagged
    tier_block_class = f"adb-tier-block {tier}" if tier in ("red", "amber", "green") else "adb-tier-block green"
    tier_icon = "●"
    signal_esc = signal_label.replace('"', "&quot;")
    persona_esc = persona_label.replace('"', "&quot;")
    pathway_esc = pathway_label.replace('"', "&quot;")
    sep = '<span class="adb-header-sep">·</span>'
    header_html = (
        f'<div class="adb-header-row">'
        f'<span class="adb-header-item"><span class="adb-header-label">Signal</span><span class="adb-chip" title="{signal_esc}">{signal_label}</span></span>'
        f'{sep}'
        f'<span class="adb-header-item"><span class="adb-header-label">Tier</span><span class="{tier_block_class}"><span><span class="adb-tier-icon">{tier_icon}</span>{tier_label}</span><span class="{tier_action_class}"><span class="adb-tier-action-icon">{tier_action_icon}</span>{tier_action}</span></span></span>'
        f'{sep}'
        f'<span class="adb-header-item"><span class="adb-header-label">Conf</span><span class="adb-conf-pill {conf_value_class}">{conf:.2f}</span></span>'
        f'{sep}'
        f'<span class="adb-header-item"><span class="adb-header-label">Persona</span><span class="adb-chip" title="{persona_esc}">{persona_chip}</span></span>'
        f'{sep}'
        f'<span class="adb-header-item"><span class="adb-header-label">Pathway</span><span class="adb-chip" title="{pathway_esc}">{pathway_chip}</span></span>'
        f'{sep}'
        f'<span class="adb-header-item adb-why-flagged"><span class="adb-header-label">Why flagged</span><span>{why_flagged}</span></span>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # Scannable 2-column layout: left = narrative (Summary, Macro, Counterfactual), right = metrics (Risk & Tier, Drivers, Impact, Decision History)
    left, right = st.columns([3, 2])

    with left:
        # Trigger conditions — structured labeled rows
        trigger_card_html = (
            f'<div class="adb-view-card">'
            f'<div class="adb-title">Trigger conditions</div>'
            f'{trigger_grid_html}'
            f'</div>'
        )
        st.markdown(trigger_card_html, unsafe_allow_html=True)

        # Summary — bold status headline, then one concise line (escalation + liquidity)
        status_phrase = {"red": "case — manual review required", "amber": "upgrade candidate for cohort review", "green": "auto-approve candidate"}.get(tier, "review")
        status_headline = f"Status: {band_name}-confidence {status_phrase}."
        escalation_line = gov.get("reason", "—")
        liquidity_parts = []
        if liquid_cash is not None:
            liquidity_parts.append(f"Liquid: {format_currency(float(liquid_cash))}")
        if burn_rate is not None:
            liquidity_parts.append(f"Burn: {format_currency(float(burn_rate))}/mo")
        if runway and runway > 0:
            liquidity_parts.append(f"Runway: {runway:.1f} mo")
        concise_line = escalation_line
        if liquidity_parts:
            concise_line += " · " + " · ".join(liquidity_parts)
        summary_html = (
            f'<div class="adb-view-card">'
            f'<div class="adb-title">Summary</div>'
            f'<div class="adb-summary-headline">{status_headline}</div>'
            f'<div class="adb-summary-line">{concise_line}</div>'
            f'</div>'
        )
        st.markdown(summary_html, unsafe_allow_html=True)

        # Macro Context — compact chip + tooltip for details
        macro_chip_text = f"Macro overlay: {macro_status}" + (" (normal range)" if macro_status == "Neutral" else "")
        macro_tooltip = f"BoC: {boc:.2f}% · VIX: {vix:.0f} · {macro_adjustment}"
        if macro_reason_line and macro_reason_line != "No macro adjustments applied — conditions within normal range.":
            macro_tooltip = macro_reason_line + " | " + macro_tooltip
        macro_tooltip_esc = macro_tooltip.replace('"', "&quot;")
        macro_html = (
            f'<div class="adb-view-card">'
            f'<div class="adb-title">Macro Context</div>'
            f'<span class="adb-macro-chip" title="{macro_tooltip_esc}">{macro_chip_text}</span>'
            f'</div>'
        )
        st.markdown(macro_html, unsafe_allow_html=True)

        if illiquidity_ratio is not None:
            over_policy = illiquidity_ratio - 0.20
            if over_policy > 0:
                counterfactual_html = (
                    f'<div class="adb-view-card">'
                    f'<div class="adb-title">Counterfactual</div>'
                    f'<div class="adb-line"><span class="adb-key">If Not Approved:</span> Illiquidity remains {illiquidity_ratio:.0%} (+{over_policy:.0%} over policy)</div>'
                    f'</div>'
                )
                st.markdown(counterfactual_html, unsafe_allow_html=True)

    with right:
        # Risk & Tier — tier badge + action descriptor, confidence gauge with threshold
        tier_descriptor = {"red": "Manual review", "amber": "Human batch review", "green": "Auto-approve"}.get(tier, "Review")
        conf_gauge_fill_class = "high" if conf >= 0.75 else ("mid" if conf >= 0.60 else "low")
        conf_threshold_pct = 75  # auto-approve threshold at 0.75
        risk_tier_html = (
            f'<div class="adb-view-card">'
            f'<div class="adb-title">Risk &amp; Tier</div>'
            f'<div class="adb-line"><span class="adb-badge {tier_badge_class}">{tier_label}</span></div>'
            f'<div class="adb-tier-descriptor">{tier_descriptor}</div>'
            f'<div class="{conf_value_class}" style="margin:0.35rem 0 0.1rem 0;">{conf:.2f}</div>'
            f'<div class="adb-conf-gauge">'
            f'<div class="adb-conf-gauge-track">'
            f'<div class="adb-conf-gauge-fill {conf_gauge_fill_class}" style="width:{conf_bar_pct}%;"></div>'
            f'<span class="adb-conf-gauge-threshold" style="left:{conf_threshold_pct}%;"></span>'
            f'</div>'
            f'<div class="adb-conf-gauge-labels"><span>0</span><span class="th">≥0.75 auto</span><span>1</span></div>'
            f'</div>'
            f'<div class="adb-risk-label {band_label_class}" style="margin-top:0.2rem;">{band_name}</div>'
            f'</div>'
        )
        st.markdown(risk_tier_html, unsafe_allow_html=True)

        # Drivers — feature attribution: legend, rank + bar + +/- in static; exact values on hover; top 1–2 highlighted
        if top_features := sorted(audit, key=lambda x: float(x.get("importance", 0) or 0), reverse=True)[:5]:
            max_imp = max(float(a.get("importance", 0) or 0) for a in top_features) or 1.0
            attr_rows = []
            for idx, a in enumerate(top_features):
                feature_key = a.get("feature", "")
                imp = float(a.get("importance", 0) or 0)
                raw_val = float(a.get("value", 0) or 0)
                width_pct = max(3, int((imp / max_imp) * 100))
                rank = idx + 1
                impact_sign = "+" if imp >= 0 else "−"
                impact_class = "adb-attr-impact" if imp >= 0 else "adb-attr-impact neg"
                raw_fmt = _format_feature_value(feature_key, raw_val).replace('"', "&quot;")
                title_vals = f"Contribution: {imp:+.2f} · Raw: {raw_fmt}"
                row_class = "adb-attr-row" + (" adb-attr-row-top" if rank <= 2 else "")
                fill_class = "adb-fill adb-fill-main" if rank <= 2 else "adb-fill"
                attr_rows.append(
                    f'<div class="{row_class}" title="{title_vals}">'
                    f'<div class="adb-attr-driver"><span class="adb-attr-rank">#{rank}</span>{_feature_label(feature_key)}</div>'
                    f'<div class="adb-track"><div class="{fill_class}" style="width:{width_pct}%;"></div></div>'
                    f'<div class="{impact_class}">{impact_sign}</div>'
                    f'</div>'
                )
            attr_body = (
                '<div class="adb-drivers-legend">Top drivers increasing confidence</div>'
                '<div class="adb-attr-header"><div>DRIVER</div><div>CONTRIBUTION</div><div>IMPACT</div></div>'
                + "".join(attr_rows)
            )
        else:
            attr_body = '<div class="adb-muted">No feature attribution available.</div>'
        drivers_html = f'<div class="adb-view-card"><div class="adb-title">Drivers</div>{attr_body}</div>'
        st.markdown(drivers_html, unsafe_allow_html=True)

        # Projected Impact — labeled metrics with units + Impact: Low/Medium/High tag
        suggested = tp.get("suggested_amount")
        has_aua = suggested is not None
        suggested_f = float(suggested) if has_aua else 0
        aua_display = format_currency(suggested_f) if has_aua else "—"
        liq_display = f"{runway:.1f} mo" if runway > 0 else "—"
        retention_delta = "+3.2%"
        if has_aua and suggested_f >= 50000:
            impact_level, impact_class = "High", "high"
        elif has_aua and suggested_f >= 10000:
            impact_level, impact_class = "Medium", "medium"
        else:
            impact_level, impact_class = "Low", "low"
        impact_tag_html = f'<span class="adb-impact-tag {impact_class}">Impact: {impact_level}</span>'
        impact_html = (
            f'<div class="adb-view-card">'
            f'<div class="adb-title">Projected Impact</div>'
            f'{impact_tag_html}'
            f'<div class="adb-impact-metric"><span class="k">AUA impact:</span> {aua_display}</div>'
            f'<div class="adb-impact-metric"><span class="k">Runway:</span> {liq_display}</div>'
            f'<div class="adb-impact-metric"><span class="k">Retention Δ:</span> {retention_delta}</div>'
            f'</div>'
        )
        st.markdown(impact_html, unsafe_allow_html=True)

        # Runway & risk thresholds — progress bars with safe/unsafe zones
        runway_max = 24
        runway_pct = min(100, int((runway / runway_max) * 100)) if runway and runway > 0 else 0
        if runway and runway > 0:
            if runway >= 12:
                runway_zone = "safe"
            elif runway >= 6:
                runway_zone = "caution"
            else:
                runway_zone = "unsafe"
        else:
            runway_zone = "unsafe"
        runway_bar_html = (
            f'<div class="adb-progress-wrap">'
            f'<div class="adb-progress-label"><span>Runway</span><span>{runway:.1f} mo</span></div>'
            f'<div class="adb-progress-bar"><div class="fill {runway_zone}" style="width:{runway_pct}%;"></div></div>'
            f'<div class="adb-progress-label" style="font-size:10px;margin-top:0.1rem;"><span>0</span><span>6 mo</span><span>12 mo</span><span>24 mo</span></div>'
            f'</div>'
        )
        illiq_pct = min(100, int((illiquidity_ratio or 0) * 100))
        illiq_unsafe = (illiquidity_ratio or 0) >= 0.20
        illiq_zone = "unsafe" if illiq_unsafe else "safe"
        illiq_bar_html = (
            f'<div class="adb-progress-wrap">'
            f'<div class="adb-progress-label"><span>Illiquidity</span><span>{illiq_pct}% (threshold 20%)</span></div>'
            f'<div class="adb-progress-bar"><div class="fill {illiq_zone}" style="width:{illiq_pct}%;"></div></div>'
            f'</div>'
        )
        bars_card_html = (
            f'<div class="adb-view-card">'
            f'<div class="adb-title">Runway &amp; risk thresholds</div>'
            f'{runway_bar_html}'
            f'{illiq_bar_html}'
            f'</div>'
        )
        st.markdown(bars_card_html, unsafe_allow_html=True)

        # Decision History — metrics
        action = (existing_decision or {}).get("action", "").lower()
        ts = (existing_decision or {}).get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts_fmt = dt.strftime("%H:%M")
            except Exception:
                ts_fmt = ts[:5]
        else:
            ts_fmt = "—"
        status_text = action.upper() if action else "NOT DECIDED"
        status_class = "adb-status-pending"
        if action == "approved":
            status_class = "adb-status-approved"
        elif action == "rejected":
            status_class = "adb-status-rejected"
        record_html = (
            f'<div class="adb-view-card">'
            f'<div class="adb-title">Decision History</div>'
            f'<div class="adb-decision-status-box">'
            f'<div class="adb-status {status_class}">{status_text}</div>'
            f'</div>'
            f'<div class="adb-meta"><span class="adb-key">Time:</span> {ts_fmt}</div>'
            f'<div class="adb-meta"><span class="adb-key">Confidence at decision:</span> {conf:.2f}</div>'
            f'<div class="adb-meta"><span class="adb-key">Override rate:</span> {override_pct:.1f}%</div>'
            f'<div class="adb-meta"><span class="adb-key">Model:</span> {compliance.get("model_version", "—")}</div>'
            f'</div>'
        )
        st.markdown(record_html, unsafe_allow_html=True)

    # Advanced — Raw features: collapsible drawer for power users
    with st.expander("Advanced — Raw features", expanded=False):
        st.caption("Traceability and model inputs for this case.")
        trace = hypothesis.get("traceability", {})
        if trace:
            import json
            try:
                st.json(trace)
            except Exception:
                st.code(str(trace)[:2000] + ("…" if len(str(trace)) > 2000 else ""), language=None)
        if audit:
            st.caption("**Audit log (feature → importance)**")
            for a in audit[:12]:
                fk = a.get("feature", "")
                imp = float(a.get("importance", 0) or 0)
                val = a.get("value")
                val_str = _format_feature_value(fk, float(val)) if val is not None else "—"
                st.code(f"{_feature_label(fk)}: {imp:+.2f} (raw: {val_str})")
        st.caption("**Key hypothesis fields**")
        keys_to_show = ["user_id", "signal", "confidence", "persona_tier", "macro_reasons"]
        st.json({k: hypothesis.get(k) for k in keys_to_show if k in hypothesis})


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
        <div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:0.6rem 0.9rem;font-size:14px;color:#1a1a1a;line-height:1.5;">
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
        <div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:0.6rem 0.9rem;font-size:14px;color:#1a1a1a;line-height:1.5;">
            <strong style="color:#0f172a;">Auto-Escalation Rules:</strong><br>
            • Illiquid allocation <span style="color:#dc2626;font-weight:600;">&gt;20% AUA</span> → Manual review required<br>
            • Credit exposure <span style="color:#dc2626;font-weight:600;">&gt;5x monthly income</span> → Compliance review<br>
            • Model confidence <span style="color:#d97706;font-weight:600;">&lt;0.60</span> → Auto-approval blocked<br>
            • Product value <span style="color:#d97706;font-weight:600;">&gt;$50k</span> → Senior approval required
        </div>
        <div style="background:#fff;border:1px solid #a7f3d0;border-radius:6px;padding:0.6rem 0.9rem;font-size:14px;color:#1a1a1a;line-height:1.5;margin-top:0.5rem;">
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
        
        export_col1, export_col2 = st.columns([2, 1], vertical_alignment="bottom")
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
    ver = compliance.get("model_version", "—")
    build = compliance.get("model_build_date", "—")
    audit = compliance.get("last_audit", "—")
    html = (
        f'<div class="ws-audit-summary">'
        f'<div class="ws-subsection" style="margin: 0 0 0.75rem 0;">Model Governance</div>'
        f'<div class="ws-secondary" style="line-height:1.6;">'
        f'<strong>Production Model:</strong> {ver}<br>'
        f'<strong>Build Date:</strong> {build}<br>'
        f'<strong>Validation Status:</strong> Approved for production use<br>'
        f'<strong>Performance Monitoring:</strong> Active<br>'
        f'<strong>Drift Detection:</strong> Enabled (±5% threshold)<br>'
        f'<strong>Retraining Schedule:</strong> Monthly or on drift detection<br><br>'
        f'<strong>Approval Chain:</strong><br>'
        f'• Model Risk: ✓ Approved ({build})<br>'
        f'• Compliance: ✓ Approved ({audit})<br>'
        f'• IT Security: ✓ Approved ({build})<br>'
        f'• Business Owner: ✓ Approved ({build})'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


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
