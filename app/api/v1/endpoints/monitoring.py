import time

from fastapi import APIRouter, Request

from app.api.schemas.response import MonitoringResponse
from app.core.config import settings

router = APIRouter()


@router.get("/monitoring", response_model=MonitoringResponse, summary="Sistem Metrikleri")
async def get_monitoring_metrics(request: Request):
    """API sunucusu çalışma süresi ve istek sayaçlarını döndürür."""
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time
    total_requests = getattr(request.app.state, "request_count", 0)

    return MonitoringResponse(
        status="healthy",
        uptime_seconds=round(uptime, 2),
        total_requests=total_requests,
        active_model=settings.MODEL_PATH,
        device=settings.DEVICE,
    )
