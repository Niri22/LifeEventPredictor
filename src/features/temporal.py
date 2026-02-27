"""Temporal feature engineering: spend velocity, savings rate, and their deltas."""

import numpy as np
import pandas as pd


def compute_temporal_features(txns: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user-month temporal features from the enriched transaction table.

    Returns a DataFrame indexed by (user_id, month) with:
      - spend_velocity_30d: total debits in chequing + credit_card
      - spend_velocity_delta: MoM change
      - savings_rate: (income - debits) / income
      - savings_rate_delta: MoM change
      - txn_count_30d: number of transactions
      - avg_txn_amount: mean absolute transaction amount
    """
    df = txns.copy()
    df["month"] = df["timestamp"].dt.to_period("M")

    # Debits: negative amounts on chequing and credit_card (exclude investment transfers, rent)
    spending_mask = (
        df["account_type"].isin(["chequing", "credit_card"])
        & (df["amount"] < 0)
        & (~df["merchant"].str.contains("CC Payment|Landlord|Mortgage", case=False, na=False))
    )
    spending = df[spending_mask].copy()
    spending["abs_amount"] = spending["amount"].abs()

    monthly_spend = (
        spending.groupby(["user_id", "month"])
        .agg(
            spend_velocity_30d=("abs_amount", "sum"),
            txn_count_30d=("txn_id", "count"),
            avg_txn_amount=("abs_amount", "mean"),
        )
        .reset_index()
    )

    # Income: positive ACH deposits to chequing (payroll)
    income_mask = (
        (df["account_type"] == "chequing")
        & (df["amount"] > 0)
        & (df["channel"] == "ach")
    )
    monthly_income = (
        df[income_mask]
        .groupby(["user_id", "month"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "monthly_income"})
    )

    result = monthly_spend.merge(monthly_income, on=["user_id", "month"], how="left")
    result["monthly_income"] = result["monthly_income"].fillna(
        result.groupby("user_id")["monthly_income"].transform("median")
    )

    result["savings_rate"] = np.where(
        result["monthly_income"] > 0,
        (result["monthly_income"] - result["spend_velocity_30d"]) / result["monthly_income"],
        0.0,
    )
    result["savings_rate"] = result["savings_rate"].clip(-1.0, 1.0)

    # Month-over-month deltas
    result = result.sort_values(["user_id", "month"])
    result["spend_velocity_delta"] = result.groupby("user_id")["spend_velocity_30d"].diff()
    result["savings_rate_delta"] = result.groupby("user_id")["savings_rate"].diff()

    result["spend_velocity_delta"] = result["spend_velocity_delta"].fillna(0.0)
    result["savings_rate_delta"] = result["savings_rate_delta"].fillna(0.0)

    return result
