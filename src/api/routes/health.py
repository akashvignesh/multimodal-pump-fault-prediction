"""Health check endpoint for pump fault risk API."""
import time

from fastapi import APIRouter

from src.api.schemas.response import HealthResponse
from src.config import settings

router = APIRouter(tags=["health"])

# Track startup time
_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health status"
)
async def health_check() -> HealthResponse:
    """Return health status with model version and uptime.
    
    Returns:
        HealthResponse: {status: "ok", model_version: "v1.0.0", uptime_s: ...}
    """
    uptime_s = round(time.time() - _start_time, 2)
    
    return HealthResponse(
        status="ok",
        model_version=settings.model_version,
        uptime_s=uptime_s
    )