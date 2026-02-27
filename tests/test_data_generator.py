"""Tests for the synthetic data engine."""

import pandas as pd
import pytest

from src.data_generator.profiles import generate_profiles
from src.data_generator.generator import generate_dataset


@pytest.fixture
def small_config():
    return {
        "data_generation": {
            "num_users": 20,
            "months": 6,
            "start_date": "2025-01-01",
            "seed": 99,
            "persona_weights": {
                "baseline": 0.40,
                "aspiring_affluent": 0.20,
                "sticky_family_leader": 0.20,
                "generation_nerd": 0.20,
            },
            "income_brackets": {
                "low": [40000, 65000],
                "mid": [65000, 120000],
                "high": [120000, 250000],
                "ultra": [250000, 500000],
            },
            "aua_ranges": {
                "aspiring_affluent": [50000, 99999],
                "sticky_family_leader": [100000, 499999],
                "generation_nerd": [500000, 1500000],
                "baseline": [5000, 49999],
            },
            "txns_per_day_lambda": 3,
            "signal_onset_buffer_months": 2,
        },
        "persona_thresholds": {
            "aspiring_affluent_min": 50000,
            "aspiring_affluent_max": 100000,
            "sticky_family_leader_min": 100000,
            "sticky_family_leader_max": 500000,
            "generation_nerd_min": 500000,
        },
        "model": {"test_size": 0.2, "random_state": 42},
    }


def test_generate_profiles(small_config):
    profiles = generate_profiles(small_config)
    assert len(profiles) == 20
    assert "user_id" in profiles.columns
    assert "persona" in profiles.columns
    assert "initial_aua" in profiles.columns
    assert "rrsp_room" in profiles.columns

    non_baseline = profiles[profiles["persona"] != "baseline"]
    assert non_baseline["signal_onset_date"].notna().all()

    baseline = profiles[profiles["persona"] == "baseline"]
    assert baseline["signal_onset_date"].isna().all()


def test_generate_dataset(small_config):
    profiles, txns = generate_dataset(small_config)
    assert len(profiles) == 20
    assert len(txns) > 0

    assert "account_type" in txns.columns
    assert "balance_after" in txns.columns

    account_types = txns["account_type"].unique()
    assert "chequing" in account_types
    assert any("investment" in at for at in account_types)


def test_opening_balances_seeded(small_config):
    _, txns = generate_dataset(small_config)
    opening = txns[txns["merchant"] == "Opening Balance"]
    assert len(opening) > 0

    invest_opening = opening[opening["account_type"].str.startswith("investment")]
    assert len(invest_opening) > 0
