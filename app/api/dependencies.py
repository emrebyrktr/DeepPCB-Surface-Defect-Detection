from fastapi import Depends, Request

from app.core.base_detector import BaseDetector
from app.services.inference_service import InferenceService


def get_detector(request: Request) -> BaseDetector:
    """FastAPI state içerisinden başlatılmış detector örneğini döndürür."""
    return request.app.state.detector


def get_inference_service(
    detector: BaseDetector = Depends(get_detector),
) -> InferenceService:
    """InferenceService nesnesini bağımlılık enjeksiyonu ile sağlar."""
    return InferenceService(detector=detector)

