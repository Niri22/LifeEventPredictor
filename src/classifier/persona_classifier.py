"""
Rule-based persona tier classifier.

Assigns each user-month observation to a wealth tier based on AUA thresholds.
This runs UPSTREAM of the ML model -- the model only predicts the signal,
not the tier.
"""

import pandas as pd

from src.utils.io import load_config

DEFAULT_THRESHOLDS = {
    "aspiring_affluent_min": 50_000,
    "aspiring_affluent_max": 100_000,
    "sticky_family_leader_min": 100_000,
    "sticky_family_leader_max": 500_000,
    "generation_nerd_min": 500_000,
}


def classify_persona_tier(
    features: pd.DataFrame, config: dict | None = None
) -> pd.DataFrame:
    """
    Add a 'persona_tier' column based on aua_current thresholds.

    Tiers:
      - not_eligible: AUA < $50k
      - aspiring_affluent: $50k <= AUA < $100k
      - sticky_family_leader: $100k <= AUA < $500k
      - generation_nerd: AUA >= $500k
    """
    if config is None:
        config = load_config()

    t = config.get("persona_thresholds", DEFAULT_THRESHOLDS)

    df = features.copy()
    conditions = [
        df["aua_current"] >= t["generation_nerd_min"],
        (df["aua_current"] >= t["sticky_family_leader_min"])
        & (df["aua_current"] < t["sticky_family_leader_max"]),
        (df["aua_current"] >= t["aspiring_affluent_min"])
        & (df["aua_current"] < t["aspiring_affluent_max"]),
    ]
    choices = ["generation_nerd", "sticky_family_leader", "aspiring_affluent"]

    df["persona_tier"] = "not_eligible"
    for cond, choice in zip(conditions, choices):
        df.loc[cond, "persona_tier"] = choice

    return df
