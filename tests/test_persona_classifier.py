"""Tests for the persona tier classifier."""

import pandas as pd
import pytest

from src.classifier.persona_classifier import classify_persona_tier


@pytest.fixture
def feature_rows():
    return pd.DataFrame([
        {"user_id": "u1", "month": "2025-01", "aua_current": 30000},
        {"user_id": "u2", "month": "2025-01", "aua_current": 75000},
        {"user_id": "u3", "month": "2025-01", "aua_current": 250000},
        {"user_id": "u4", "month": "2025-01", "aua_current": 750000},
    ])


def test_tier_assignment(feature_rows):
    result = classify_persona_tier(feature_rows)
    assert result.loc[result["user_id"] == "u1", "persona_tier"].iloc[0] == "not_eligible"
    assert result.loc[result["user_id"] == "u2", "persona_tier"].iloc[0] == "aspiring_affluent"
    assert result.loc[result["user_id"] == "u3", "persona_tier"].iloc[0] == "sticky_family_leader"
    assert result.loc[result["user_id"] == "u4", "persona_tier"].iloc[0] == "generation_nerd"


def test_boundary_values():
    df = pd.DataFrame([
        {"user_id": "edge1", "month": "2025-01", "aua_current": 50000},
        {"user_id": "edge2", "month": "2025-01", "aua_current": 99999.99},
        {"user_id": "edge3", "month": "2025-01", "aua_current": 100000},
        {"user_id": "edge4", "month": "2025-01", "aua_current": 500000},
    ])
    result = classify_persona_tier(df)
    tiers = result.set_index("user_id")["persona_tier"]
    assert tiers["edge1"] == "aspiring_affluent"
    assert tiers["edge2"] == "aspiring_affluent"
    assert tiers["edge3"] == "sticky_family_leader"
    assert tiers["edge4"] == "generation_nerd"
