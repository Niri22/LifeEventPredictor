"""Tests for the FastAPI Signal Engine endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Wealthsimple Pulse" in r.json()["service"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert len(data["models_loaded"]) == 3


def test_predict_empty_transactions():
    r = client.post("/predict", json={
        "user_id": "test-user",
        "rrsp_room": 25000,
        "transactions": [],
    })
    assert r.status_code == 400


def test_predict_minimal():
    """Smoke test with a handful of transactions."""
    import uuid
    from datetime import datetime, timedelta

    txns = []
    base = datetime(2025, 6, 1)
    for i in range(30):
        dt = base + timedelta(days=i)
        # Payroll on 1st
        if i == 0:
            txns.append({
                "txn_id": str(uuid.uuid4()),
                "timestamp": dt.isoformat(),
                "account_type": "chequing",
                "amount": 8000.0,
                "merchant": "Employer Payroll",
                "mcc": 6012,
                "mcc_category": "financial_services",
                "balance_after": 80000.0,
                "channel": "ach",
            })
            txns.append({
                "txn_id": str(uuid.uuid4()),
                "timestamp": dt.isoformat(),
                "account_type": "investment_rrsp",
                "amount": 60000.0,
                "merchant": "Opening Balance",
                "mcc": 6012,
                "mcc_category": "financial_services",
                "balance_after": 60000.0,
                "channel": "ach",
            })
        txns.append({
            "txn_id": str(uuid.uuid4()),
            "timestamp": dt.isoformat(),
            "account_type": "chequing",
            "amount": -50.0,
            "merchant": "Coffee Shop",
            "mcc": 5812,
            "mcc_category": "dining",
            "balance_after": 70000.0 - i * 50,
            "channel": "pos",
        })

    r = client.post("/predict", json={
        "user_id": "smoke-test-user",
        "rrsp_room": 30000,
        "transactions": txns,
    })
    assert r.status_code == 200
    data = r.json()
    assert "persona_tier" in data
    assert "user_id" in data
