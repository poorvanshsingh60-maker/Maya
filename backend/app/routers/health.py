"""
Health-check router.
"""
from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.services.model_service import ModelService

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    service = ModelService.get_instance()
    return HealthResponse(
        status="ok" if service.is_loaded else "unavailable",
        model_loaded=service.is_loaded,
        device=str(service.device) if service.device else "unknown",
    )
