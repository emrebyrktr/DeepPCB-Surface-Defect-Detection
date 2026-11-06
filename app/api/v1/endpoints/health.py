from fastapi import APIRouter, Depends

from app.api.dependencies import get_detector
from app.api.schemas.response import HealthResponse
from app.core.base_detector import BaseDetector
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Sağlık Kontrolü")
async def health_check(detector: BaseDetector = Depends(get_detector)):
    """API ve model durumunun sağlıklı olup olmadığını doğrular."""
    return HealthResponse(
        status="healthy",
        model_name="DeepPCB-YOLOv8",
        model_version=settings.VERSION,
        device=settings.DEVICE,
        classes=detector.classes,
    )

