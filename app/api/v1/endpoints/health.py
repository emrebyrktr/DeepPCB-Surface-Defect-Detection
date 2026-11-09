from fastapi import APIRouter, Depends

from app.api.dependencies import get_detector
from app.api.schemas.response import HealthResponse
from app.core.base_detector import BaseDetector

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Sağlık Kontrolü")
async def health_check(detector: BaseDetector = Depends(get_detector)):
    """Yüklenmiş modelin kimliğini ve servis durumunu döndürür."""
    return HealthResponse(
        status="healthy",
        model_name=detector.model_name,
        model_version=detector.model_version,
        device=detector.device,
        classes=detector.classes,
    )

