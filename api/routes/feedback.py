"""POST /feedback -- record curator decisions. POST /batch/approve -- batch approve cohort."""

from fastapi import APIRouter

from api.schemas import BatchApproveRequest, BatchApproveResponse, FeedbackRequest, FeedbackResponse
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


@router.post("/batch/approve", response_model=BatchApproveResponse)
def batch_approve(req: BatchApproveRequest):
    """Record approval (or rejection) for a list of cohort members."""
    for item in req.items:
        record_feedback(
            user_id=item.user_id,
            persona_tier=item.persona_tier,
            signal=item.signal,
            product_code=item.product_code,
            confidence=item.confidence,
            governance_tier=item.governance_tier,
            action=req.action,
        )
    return BatchApproveResponse(
        approved_count=len(req.items),
        status="ok",
    )


@router.get("/feedback/stats")
def feedback_stats():
    return get_feedback_stats()
