"""Wealth-tier feature engineering: AUA trajectory, RRSP utilization, illiquidity ratio."""

import numpy as np
import pandas as pd


def compute_wealth_features(
    txns: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute per-user-month wealth-specific features.

    Returns a DataFrame indexed by (user_id, month) with:
      - aua_current: total AUA at month-end (sum of investment account balances)
      - aua_delta: month-over-month AUA change
      - rrsp_utilization: cumulative RRSP deposits / total RRSP room
      - illiquidity_ratio: non-reg investment balance / total AUA
      - credit_spend_vs_invest: credit card debits / investment transfers
    """
    df = txns.copy()
    df["month"] = df["timestamp"].dt.to_period("M")

    investment_types = [
        "investment_rrsp", "investment_tfsa",
        "investment_resp", "investment_non_reg",
    ]

    # AUA at month-end: last balance_after per investment account per month
    invest_txns = df[df["account_type"].isin(investment_types)].copy()
    invest_txns = invest_txns.sort_values("timestamp")

    month_end_balances = (
        invest_txns.groupby(["user_id", "month", "account_type"])["balance_after"]
        .last()
        .reset_index()
    )
    aua_by_month = (
        month_end_balances.groupby(["user_id", "month"])["balance_after"]
        .sum()
        .reset_index()
        .rename(columns={"balance_after": "aua_current"})
    )
    aua_by_month = aua_by_month.sort_values(["user_id", "month"])
    aua_by_month["aua_delta"] = aua_by_month.groupby("user_id")["aua_current"].diff().fillna(0.0)

    # RRSP utilization: cumulative positive RRSP transfers / total RRSP room
    rrsp_deposits = (
        invest_txns[
            (invest_txns["account_type"] == "investment_rrsp") & (invest_txns["amount"] > 0)
        ]
        .groupby(["user_id", "month"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "rrsp_deposit_month"})
    )
    rrsp_deposits = rrsp_deposits.sort_values(["user_id", "month"])
    rrsp_deposits["cumulative_rrsp"] = rrsp_deposits.groupby("user_id")["rrsp_deposit_month"].cumsum()

    rrsp_room = profiles[["user_id", "rrsp_room"]].copy()
    rrsp_deposits = rrsp_deposits.merge(rrsp_room, on="user_id", how="left")
    rrsp_deposits["rrsp_utilization"] = np.where(
        rrsp_deposits["rrsp_room"] > 0,
        rrsp_deposits["cumulative_rrsp"] / (rrsp_deposits["rrsp_room"] + rrsp_deposits["cumulative_rrsp"]),
        0.0,
    )
    rrsp_util = rrsp_deposits[["user_id", "month", "rrsp_utilization"]]

    # Illiquidity ratio: non-reg balance / total AUA
    non_reg_balance = (
        month_end_balances[month_end_balances["account_type"] == "investment_non_reg"]
        [["user_id", "month", "balance_after"]]
        .rename(columns={"balance_after": "non_reg_balance"})
    )
    illiquidity = aua_by_month.merge(non_reg_balance, on=["user_id", "month"], how="left")
    illiquidity["non_reg_balance"] = illiquidity["non_reg_balance"].fillna(0.0)
    illiquidity["illiquidity_ratio"] = np.where(
        illiquidity["aua_current"] > 0,
        illiquidity["non_reg_balance"] / illiquidity["aua_current"],
        0.0,
    )

    # Credit spend vs invest: monthly credit card debits / monthly investment transfers
    cc_spend = (
        df[(df["account_type"] == "credit_card") & (df["amount"] < 0)]
        .groupby(["user_id", "month"])["amount"]
        .sum()
        .abs()
        .reset_index()
        .rename(columns={"amount": "cc_spend_30d"})
    )
    invest_transfers = (
        df[
            (df["account_type"].isin(investment_types))
            & (df["amount"] > 0)
            & (df["channel"] == "internal_transfer")
        ]
        .groupby(["user_id", "month"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "invest_inflow"})
    )
    ratio = cc_spend.merge(invest_transfers, on=["user_id", "month"], how="outer")
    ratio["cc_spend_30d"] = ratio["cc_spend_30d"].fillna(0.0)
    ratio["invest_inflow"] = ratio["invest_inflow"].fillna(0.0)
    ratio["credit_spend_vs_invest"] = np.where(
        ratio["invest_inflow"] > 0,
        ratio["cc_spend_30d"] / ratio["invest_inflow"],
        0.0,
    )

    # Merge all wealth features
    result = aua_by_month[["user_id", "month", "aua_current", "aua_delta"]]
    result = result.merge(rrsp_util, on=["user_id", "month"], how="left")
    result = result.merge(
        illiquidity[["user_id", "month", "illiquidity_ratio"]],
        on=["user_id", "month"], how="left",
    )
    result = result.merge(
        ratio[["user_id", "month", "credit_spend_vs_invest", "cc_spend_30d"]],
        on=["user_id", "month"], how="left",
    )

    result["rrsp_utilization"] = result["rrsp_utilization"].fillna(0.0)
    result["illiquidity_ratio"] = result["illiquidity_ratio"].fillna(0.0)
    result["credit_spend_vs_invest"] = result["credit_spend_vs_invest"].fillna(0.0)
    result["cc_spend_30d"] = result["cc_spend_30d"].fillna(0.0)

    # Distance-to-upgrade (milestone) features
    aua = result["aua_current"]
    result["gap_to_next_milestone"] = 0.0
    result["pct_to_milestone"] = 1.0
    result["next_milestone_name"] = "At cap"
    mask_below_premium = aua < 100_000
    result.loc[mask_below_premium, "gap_to_next_milestone"] = 100_000 - aua[mask_below_premium]
    result.loc[mask_below_premium, "pct_to_milestone"] = (aua[mask_below_premium] / 100_000).clip(0, 1)
    result.loc[mask_below_premium, "next_milestone_name"] = "Premium ($100k)"
    mask_below_legacy = (aua >= 100_000) & (aua < 500_000)
    result.loc[mask_below_legacy, "gap_to_next_milestone"] = 500_000 - aua[mask_below_legacy]
    result.loc[mask_below_legacy, "pct_to_milestone"] = (aua[mask_below_legacy] / 500_000).clip(0, 1)
    result.loc[mask_below_legacy, "next_milestone_name"] = "Legacy ($500k)"

    return result
