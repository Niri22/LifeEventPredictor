"""Single-user inference wrapper: loads per-persona models and returns signal hypotheses."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[2] / "data"


class BaseSignalModel(ABC):
    """Abstract interface for signal detection -- swap in TCN later."""

    @abstractmethod
    def predict(self, features: dict, persona_tier: str) -> dict:
        ...


class XGBSignalModel(BaseSignalModel):
    """XGBoost-based signal detector using per-persona serialized models."""

    def __init__(self):
        self._models: dict[str, dict] = {}
        self._loaded = False

    def load_models(self) -> None:
        for persona in ["aspiring_affluent", "sticky_family_leader", "generation_nerd"]:
            path = MODEL_DIR / f"model_{persona}.joblib"
            if path.exists():
                self._models[persona] = joblib.load(path)
        self._loaded = True

    def predict(self, features: dict, persona_tier: str) -> dict | None:
        if not self._loaded:
            self.load_models()

        if persona_tier not in self._models:
            return None

        artifact = self._models[persona_tier]
        model = artifact["model"]
        feat_names = artifact["features"]
        target_label = artifact["target_label"]
        threshold = artifact["metrics"].get("optimal_threshold", 0.5)

        X = np.array([[features.get(f, 0.0) for f in feat_names]])
        proba = model.predict_proba(X)[0, 1]
        is_signal = proba >= threshold

        if not is_signal:
            return None

        return {
            "signal": target_label,
            "confidence": round(float(proba), 3),
            "contributing_features": _build_audit_log(features, feat_names, model),
        }


PRODUCT_MAP = {
    "aspiring_affluent": {
        "code": "RRSP_LOAN",
        "name": "Retirement Accelerator (RRSP Loan)",
    },
    "sticky_family_leader": {
        "code": "SUMMIT_PORTFOLIO",
        "name": "Summit Portfolio (Private Equity) + WS Credit Card",
    },
    "generation_nerd": {
        "code": "AI_RESEARCH_DIRECT_INDEX",
        "name": "AI Research Dashboard + Direct Indexing + Private Credit",
    },
}


def predict_signal(features: dict, persona_tier: str, model: BaseSignalModel) -> dict | None:
    """
    Run inference for a single user-month and build the full hypothesis payload.
    Returns None if no signal is detected.
    """
    result = model.predict(features, persona_tier)
    if result is None:
        return None

    product = PRODUCT_MAP.get(persona_tier, {})

    # Build traceability
    aua = features.get("aua_current", 0)
    spend_30d = features.get("spend_velocity_30d", 0)
    monthly_income = spend_30d / max(1 - features.get("savings_rate", 0.2), 0.1)
    liquid_cash = max(aua * 0.1, 0)  # simplified estimate

    hypothesis = {
        "persona_tier": persona_tier,
        "signal": result["signal"],
        "confidence": result["confidence"],
        "traceability": {
            "spending_buffer": {
                "liquid_cash": round(liquid_cash, 2),
                "monthly_burn_rate": round(spend_30d, 2),
                "months_of_runway": round(liquid_cash / max(spend_30d, 1), 1),
            },
            "target_product": {
                "code": product.get("code", "UNKNOWN"),
                "name": product.get("name", "Unknown Product"),
            },
            "audit_log": result["contributing_features"],
        },
        "staged_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending_review",
    }

    # Persona-specific enrichment
    if persona_tier == "aspiring_affluent":
        rrsp_util = features.get("rrsp_utilization", 0)
        rrsp_room_est = aua / max(rrsp_util, 0.01) * (1 - rrsp_util) if rrsp_util > 0 else 25000
        suggested = min(rrsp_room_est * 0.9, 100_000 - aua)
        hypothesis["traceability"]["target_product"]["suggested_amount"] = round(max(suggested, 0), 2)
        hypothesis["traceability"]["target_product"]["projected_yield"] = "Premium status unlock ($100k+ AUA)"
    elif persona_tier == "sticky_family_leader":
        hypothesis["traceability"]["target_product"]["projected_yield"] = (
            "Institutional PE access with liquidity safeguard"
        )
    elif persona_tier == "generation_nerd":
        hypothesis["traceability"]["target_product"]["projected_yield"] = (
            "Tax-optimized returns via direct indexing + AI research"
        )

    return hypothesis


def _build_audit_log(features: dict, feat_names: list[str], model) -> list[dict]:
    """Build the audit log from feature importances and values."""
    importances = model.feature_importances_
    entries = []
    for fname, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        entries.append({
            "feature": fname,
            "value": round(features.get(fname, 0.0), 4),
            "importance": round(float(imp), 4),
        })
    return entries
