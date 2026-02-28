"""
Training entry point: per-persona binary signal detectors.

For each persona tier, trains a binary classifier that detects whether a user
is exhibiting the target signal (leapfrog_ready, liquidity_warning, harvest_opportunity).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.classifier.persona_classifier import classify_persona_tier
from src.utils.io import DATA_PROCESSED, load_config, read_parquet

MODEL_DIR = Path(__file__).resolve().parents[2] / "data"

# Features used per persona (most important first)
PERSONA_FEATURES = {
    "aspiring_affluent": [
        "rrsp_utilization", "savings_rate", "savings_rate_delta",
        "aua_current", "aua_delta", "spend_velocity_30d",
        "spend_velocity_delta", "mcc_entropy", "txn_count_30d",
    ],
    "sticky_family_leader": [
        "illiquidity_ratio", "credit_spend_vs_invest", "spend_velocity_delta",
        "aua_current", "aua_delta", "mcc_entropy", "top_mcc_concentration",
        "spend_velocity_30d", "savings_rate",
    ],
    "generation_nerd": [
        "aua_current", "aua_delta", "txn_count_30d", "mcc_entropy",
        "credit_spend_vs_invest", "spend_velocity_30d", "spend_velocity_delta",
        "savings_rate", "avg_txn_amount",
    ],
}

PERSONA_LABELS = {
    "aspiring_affluent": "leapfrog_ready",
    "sticky_family_leader": "liquidity_warning",
    "generation_nerd": "harvest_opportunity",
}


def _split_by_user(df: pd.DataFrame, test_size: float, seed: int):
    """Split by user_id to prevent data leakage."""
    users = df["user_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    split_idx = int(len(users) * (1 - test_size))
    train_users = set(users[:split_idx])
    test_users = set(users[split_idx:])
    return df[df["user_id"].isin(train_users)], df[df["user_id"].isin(test_users)]


def train_all_models(config: dict | None = None) -> dict:
    """Train one binary classifier per persona tier. Returns dict of fitted models."""
    if config is None:
        config = load_config()

    model_cfg = config["model"]
    features = read_parquet(DATA_PROCESSED / "features.parquet")
    features = classify_persona_tier(features, config)

    models = {}

    for persona, target_label in PERSONA_LABELS.items():
        print(f"\n{'='*60}")
        print(f"Training model for: {persona} (detecting '{target_label}')")
        print(f"{'='*60}")

        # Include all rows where the user falls into this tier (both pre- and post-onset)
        tier_data = features[features["persona_tier"] == persona].copy()

        if len(tier_data) == 0:
            print(f"  SKIP: No data for {persona}")
            continue

        tier_data["target"] = (tier_data["label"] == target_label).astype(int)

        # If severely imbalanced (all one class), augment with baseline "none" samples
        if tier_data["target"].nunique() < 2:
            none_samples = features[
                (features["label"] == "none") & (features["persona_tier"] != "not_eligible")
            ].sample(n=min(len(tier_data), len(features[features["label"] == "none"])),
                     random_state=model_cfg["random_state"])
            none_samples = none_samples.copy()
            none_samples["target"] = 0
            tier_data = pd.concat([tier_data, none_samples], ignore_index=True)
        feat_cols = PERSONA_FEATURES[persona]

        print(f"  Samples: {len(tier_data)} | Positive: {tier_data['target'].sum()} | "
              f"Negative: {(~tier_data['target'].astype(bool)).sum()}")

        train_df, test_df = _split_by_user(tier_data, model_cfg["test_size"], model_cfg["random_state"])

        X_train = train_df[feat_cols].values
        y_train = train_df["target"].values
        X_test = test_df[feat_cols].values
        y_test = test_df["target"].values

        # Scale weight for class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale = n_neg / max(n_pos, 1)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale,
            random_state=model_cfg["random_state"],
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Evaluate
        from src.models.evaluate import evaluate_model
        metrics = evaluate_model(model, X_test, y_test, persona)
        classification_report_str = metrics.pop("classification_report", None)

        # Save model + metadata
        artifact = {
            "model": model,
            "features": feat_cols,
            "persona": persona,
            "target_label": target_label,
            "metrics": metrics,
        }
        if classification_report_str:
            artifact["classification_report"] = classification_report_str
        path = MODEL_DIR / f"model_{persona}.joblib"
        joblib.dump(artifact, path)
        print(f"  Saved: {path}")

        models[persona] = artifact

    return models


if __name__ == "__main__":
    train_all_models()
