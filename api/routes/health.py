"""GET /health -- service health and model status."""

from fastapi import APIRouter

from api.schemas import HealthResponse
from api.routes.predict import get_model

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    model = get_model()
    loaded = list(model._models.keys())
    return HealthResponse(
        status="healthy",
        models_loaded=loaded,
        version="0.1.0",
    )
