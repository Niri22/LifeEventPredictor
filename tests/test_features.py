"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.temporal import compute_temporal_features
from src.features.categorical import compute_categorical_features
from src.features.wealth import compute_wealth_features


@pytest.fixture
def sample_txns():
    """Minimal transaction set for two users over two months."""
    records = []
    for uid in ["user_a", "user_b"]:
        for month in [1, 2]:
            # Payroll
            records.append({
                "txn_id": f"{uid}_pay_{month}",
                "user_id": uid,
                "timestamp": pd.Timestamp(f"2025-0{month}-01"),
                "account_type": "chequing",
                "amount": 5000.0,
                "merchant": "Employer Payroll",
                "mcc": 6012,
                "mcc_category": "financial_services",
                "balance_after": 10000.0,
                "channel": "ach",
            })
            # Spend
            for i, cat in enumerate(["groceries", "dining", "transport"]):
                records.append({
                    "txn_id": f"{uid}_spend_{month}_{i}",
                    "user_id": uid,
                    "timestamp": pd.Timestamp(f"2025-0{month}-{10+i:02d}"),
                    "account_type": "chequing",
                    "amount": -(100.0 + i * 50),
                    "merchant": f"Store_{cat}",
                    "mcc": 5411 + i,
                    "mcc_category": cat,
                    "balance_after": 9000.0 - i * 50,
                    "channel": "pos",
                })
            # Investment transfer
            records.append({
                "txn_id": f"{uid}_invest_{month}",
                "user_id": uid,
                "timestamp": pd.Timestamp(f"2025-0{month}-02"),
                "account_type": "investment_rrsp",
                "amount": 500.0,
                "merchant": "Wealthsimple Transfer",
                "mcc": 6012,
                "mcc_category": "financial_services",
                "balance_after": 50000.0 + month * 500,
                "channel": "internal_transfer",
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_profiles():
    return pd.DataFrame([
        {"user_id": "user_a", "rrsp_room": 30000.0, "annual_income": 80000, "persona": "baseline", "signal_onset_date": None},
        {"user_id": "user_b", "rrsp_room": 50000.0, "annual_income": 120000, "persona": "baseline", "signal_onset_date": None},
    ])


def test_temporal_features(sample_txns, sample_profiles):
    result = compute_temporal_features(sample_txns, sample_profiles)
    assert "spend_velocity_30d" in result.columns
    assert "savings_rate" in result.columns
    assert len(result) > 0
    assert (result["spend_velocity_30d"] >= 0).all()


def test_categorical_features(sample_txns):
    result = compute_categorical_features(sample_txns)
    assert "mcc_entropy" in result.columns
    assert "top_mcc_concentration" in result.columns
    assert (result["mcc_entropy"] >= 0).all()
    assert (result["top_mcc_concentration"] >= 0).all()
    assert (result["top_mcc_concentration"] <= 1).all()


def test_wealth_features(sample_txns, sample_profiles):
    result = compute_wealth_features(sample_txns, sample_profiles)
    assert "aua_current" in result.columns
    assert "rrsp_utilization" in result.columns
    assert len(result) > 0
