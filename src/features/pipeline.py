"""Full feature engineering pipeline: temporal + categorical + wealth → features.parquet."""

import pandas as pd

from src.features.categorical import compute_categorical_features
from src.features.temporal import compute_temporal_features
from src.features.wealth import compute_wealth_features
from src.utils.io import DATA_PROCESSED, DATA_RAW, read_parquet, write_parquet


def build_features(
    txns: pd.DataFrame | None = None,
    profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the complete feature matrix from raw data."""
    if txns is None:
        txns = read_parquet(DATA_RAW / "transactions.parquet")
    if profiles is None:
        profiles = read_parquet(DATA_RAW / "user_profiles.parquet")

    txns["timestamp"] = pd.to_datetime(txns["timestamp"])

    print("Computing temporal features...")
    temporal = compute_temporal_features(txns, profiles)

    print("Computing categorical features...")
    categorical = compute_categorical_features(txns)

    print("Computing wealth features...")
    wealth = compute_wealth_features(txns, profiles)

    print("Merging feature sets...")
    features = temporal.merge(categorical, on=["user_id", "month"], how="outer")
    features = features.merge(wealth, on=["user_id", "month"], how="outer")

    # Fill any remaining NaN from outer joins
    numeric_cols = features.select_dtypes(include="number").columns
    features[numeric_cols] = features[numeric_cols].fillna(0.0)

    # Attach ground-truth labels from profiles
    profile_labels = profiles[["user_id", "persona", "signal_onset_date"]].copy()
    features = features.merge(profile_labels, on="user_id", how="left")

    # Label: persona name if month >= onset month, else "none"
    features["signal_onset_date"] = pd.to_datetime(features["signal_onset_date"])
    features["onset_month"] = features["signal_onset_date"].dt.to_period("M")

    features["label"] = "none"
    has_onset = features["onset_month"].notna()
    after_onset = has_onset & (features["month"] >= features["onset_month"])

    label_map = {
        "aspiring_affluent": "leapfrog_ready",
        "sticky_family_leader": "liquidity_warning",
        "generation_nerd": "harvest_opportunity",
    }
    for persona, label in label_map.items():
        mask = after_onset & (features["persona"] == persona)
        features.loc[mask, "label"] = label

    features = features.drop(columns=["signal_onset_date", "onset_month", "persona"])

    # Convert Period to string for parquet compatibility
    features["month"] = features["month"].astype(str)

    return features


def run_pipeline() -> pd.DataFrame:
    """Run the full pipeline and write results to disk."""
    features = build_features()
    print(f"Writing {len(features):,} feature rows to {DATA_PROCESSED}...")
    write_parquet(features, DATA_PROCESSED / "features.parquet")
    print(f"Done. Label distribution:\n{features['label'].value_counts().to_string()}")
    return features


if __name__ == "__main__":
    run_pipeline()
