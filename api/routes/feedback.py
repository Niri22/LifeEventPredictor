"""POST /feedback -- record curator decisions for active learning."""

from fastapi import APIRouter

from api.schemas import FeedbackRequest, FeedbackResponse
from src.api.feedback import get_feedback_stats, record_feedback

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    record_feedback(
        user_id=req.user_id,
        persona_tier=req.persona_tier,
        signal=req.signal,
        product_code=req.product_code,
        confidence=req.confidence,
        governance_tier=req.governance_tier,
        action=req.action,
        reason=req.reason,
    )
    stats = get_feedback_stats()
    return FeedbackResponse(status="recorded", total_feedback=stats["total"])


@router.get("/feedback/stats")
def feedback_stats():
    return get_feedback_stats()
