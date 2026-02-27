"""Categorical feature engineering: MCC entropy, concentration, new-MCC flags."""

import numpy as np
import pandas as pd
from scipy.stats import entropy


def compute_categorical_features(txns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user-month categorical features.

    Returns a DataFrame indexed by (user_id, month) with:
      - mcc_entropy: Shannon entropy of MCC category distribution
      - top_mcc_concentration: fraction of spend in the top MCC category
      - new_mcc_flag: 1 if a new MCC category appeared this month for the first time
    """
    df = txns.copy()
    df["month"] = df["timestamp"].dt.to_period("M")

    spending_mask = df["amount"] < 0
    spend = df[spending_mask].copy()
    spend["abs_amount"] = spend["amount"].abs()

    # MCC category distribution per user-month
    cat_spend = (
        spend.groupby(["user_id", "month", "mcc_category"])["abs_amount"]
        .sum()
        .reset_index()
    )

    def _entropy_and_concentration(group: pd.DataFrame) -> pd.Series:
        total = group["abs_amount"].sum()
        if total == 0:
            return pd.Series({"mcc_entropy": 0.0, "top_mcc_concentration": 0.0})
        probs = (group["abs_amount"] / total).values
        return pd.Series({
            "mcc_entropy": float(entropy(probs, base=2)),
            "top_mcc_concentration": float(probs.max()),
        })

    features = (
        cat_spend.groupby(["user_id", "month"])
        .apply(_entropy_and_concentration, include_groups=False)
        .reset_index()
    )

    # New MCC flag: detect first appearance of any MCC category per user
    all_months = spend.groupby(["user_id", "month"])["mcc_category"].apply(set).reset_index()
    all_months = all_months.sort_values(["user_id", "month"])

    new_flags = []
    seen: dict[str, set] = {}
    for _, row in all_months.iterrows():
        uid = row["user_id"]
        cats = row["mcc_category"]
        if uid not in seen:
            seen[uid] = set()
            new_flags.append(0)
        else:
            new_cats = cats - seen[uid]
            new_flags.append(1 if new_cats else 0)
        seen[uid].update(cats)

    all_months["new_mcc_flag"] = new_flags
    all_months = all_months.drop(columns=["mcc_category"])

    features = features.merge(all_months, on=["user_id", "month"], how="left")
    features["new_mcc_flag"] = features["new_mcc_flag"].fillna(0).astype(int)

    return features
