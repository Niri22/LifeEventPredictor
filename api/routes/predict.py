"""POST /predict -- persona-routed signal detection endpoint."""

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    AuditEntry,
    PredictRequest,
    PredictResponse,
    PersonaTier,
    SignalHypothesis,
    SpendingBuffer,
    TargetProduct,
    Traceability,
)
from src.classifier.persona_classifier import classify_persona_tier
from src.features.pipeline import build_features
from src.models.predict import XGBSignalModel, predict_signal

router = APIRouter()

_model = XGBSignalModel()


def get_model() -> XGBSignalModel:
    if not _model._loaded:
        _model.load_models()
    return _model


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = get_model()

    txn_records = [
        {
            "txn_id": t.txn_id,
            "user_id": request.user_id,
            "timestamp": t.timestamp,
            "account_type": t.account_type.value,
            "amount": t.amount,
            "merchant": t.merchant,
            "mcc": t.mcc,
            "mcc_category": t.mcc_category,
            "balance_after": t.balance_after,
            "channel": t.channel.value,
        }
        for t in request.transactions
    ]

    if not txn_records:
        raise HTTPException(status_code=400, detail="No transactions provided")

    txns_df = pd.DataFrame(txn_records)
    txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])

    # Build a minimal profile for feature engineering
    profiles_df = pd.DataFrame([{
        "user_id": request.user_id,
        "rrsp_room": request.rrsp_room,
        "persona": "unknown",
        "signal_onset_date": None,
        "annual_income": 0,
    }])

    try:
        features = build_features(txns_df, profiles_df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    if features.empty:
        return PredictResponse(
            user_id=request.user_id,
            persona_tier="not_eligible",
            message="Insufficient data to compute features",
        )

    features = classify_persona_tier(features)
    latest = features.sort_values("month").iloc[-1]
    persona_tier = latest["persona_tier"]

    if persona_tier == "not_eligible":
        return PredictResponse(
            user_id=request.user_id,
            persona_tier=persona_tier,
            message="AUA below $50k threshold -- not eligible for product recommendations",
        )

    feature_dict = latest.to_dict()
    hypothesis_raw = predict_signal(feature_dict, persona_tier, model)

    if hypothesis_raw is None:
        return PredictResponse(
            user_id=request.user_id,
            persona_tier=persona_tier,
            message=f"No signal detected for {persona_tier} tier at this time",
        )

    trace = hypothesis_raw["traceability"]
    hypothesis = SignalHypothesis(
        user_id=request.user_id,
        persona_tier=PersonaTier(persona_tier),
        signal=hypothesis_raw["signal"],
        confidence=hypothesis_raw["confidence"],
        traceability=Traceability(
            spending_buffer=SpendingBuffer(**trace["spending_buffer"]),
            target_product=TargetProduct(**trace["target_product"]),
            audit_log=[AuditEntry(**e) for e in trace["audit_log"]],
        ),
        staged_at=hypothesis_raw["staged_at"],
    )

    return PredictResponse(
        user_id=request.user_id,
        persona_tier=persona_tier,
        hypothesis=hypothesis,
        message=f"Signal detected: {hypothesis_raw['signal']} (confidence: {hypothesis_raw['confidence']})",
    )
