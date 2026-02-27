"""POST /predict -- persona-routed signal detection with governance, macro, nudges, and feedback."""

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    AuditEntry,
    GovernanceTier,
    MacroContext,
    PredictRequest,
    PredictResponse,
    PersonaTier,
    SignalHypothesis,
    SpendingBuffer,
    TargetProduct,
    Traceability,
)
from src.api.feedback import apply_feedback_penalty
from src.api.macro_agent import adjust_confidence_for_macro, fetch_macro_snapshot
from src.classifier.persona_classifier import classify_persona_tier
from src.features.nudges import generate_composite_reason, generate_nudge
from src.features.pipeline import build_features
from src.models.governance import classify_governance_tier
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
            user_id=request.user_id, persona_tier="not_eligible",
            message="Insufficient data to compute features",
        )

    features = classify_persona_tier(features)
    latest = features.sort_values("month").iloc[-1]
    persona_tier = latest["persona_tier"]

    if persona_tier == "not_eligible":
        return PredictResponse(
            user_id=request.user_id, persona_tier=persona_tier,
            message="AUA below $50k threshold",
        )

    feature_dict = latest.to_dict()
    hypothesis_raw = predict_signal(feature_dict, persona_tier, model)

    if hypothesis_raw is None:
        return PredictResponse(
            user_id=request.user_id, persona_tier=persona_tier,
            message=f"No signal detected for {persona_tier}",
        )

    trace = hypothesis_raw["traceability"]
    product_code = trace["target_product"]["code"]
    product_name = trace["target_product"]["name"]

    # Macro adjustment
    macro = fetch_macro_snapshot()
    adj_conf, macro_reasons = adjust_confidence_for_macro(
        hypothesis_raw["confidence"], persona_tier, product_code, macro,
    )

    # Feedback penalty
    fb_conf, fb_reason = apply_feedback_penalty(adj_conf, persona_tier, hypothesis_raw["signal"], product_code)

    # Governance tier
    illiquidity = 0.0
    for e in trace["audit_log"]:
        if e["feature"] == "illiquidity_ratio":
            illiquidity = e["value"]
            break
    gov = classify_governance_tier(fb_conf, product_code, illiquidity)

    # Nudge
    behavioral = generate_nudge(persona_tier, hypothesis_raw["signal"], feature_dict, product_name)
    nudge = generate_composite_reason(behavioral, macro_reasons, fb_reason)

    hypothesis = SignalHypothesis(
        user_id=request.user_id,
        persona_tier=PersonaTier(persona_tier),
        signal=hypothesis_raw["signal"],
        confidence=fb_conf,
        traceability=Traceability(
            spending_buffer=SpendingBuffer(**trace["spending_buffer"]),
            target_product=TargetProduct(**trace["target_product"]),
            audit_log=[AuditEntry(**e) for e in trace["audit_log"]],
        ),
        governance=GovernanceTier(**gov),
        macro_context=MacroContext(
            boc_prime_rate=macro.boc_prime_rate,
            vix=macro.vix,
            tsx_volatility=macro.tsx_volatility,
            rates_high=macro.rates_high,
            market_volatile=macro.market_volatile,
        ),
        macro_reasons=macro_reasons,
        nudge=nudge,
        feedback_reason=fb_reason,
        staged_at=hypothesis_raw["staged_at"],
    )

    return PredictResponse(
        user_id=request.user_id,
        persona_tier=persona_tier,
        hypothesis=hypothesis,
        message=f"Signal: {hypothesis_raw['signal']} | Gov: {gov['tier']} | Conf: {fb_conf}",
    )
