from typing import Any

from fastapi import HTTPException, UploadFile, status

from app.core.base_detector import BaseDetector
from app.services.feature_pipeline import ImageFeaturePipeline

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class InferenceService:
    """Çıkarım iş akışını yöneten servis katmanı."""

    def __init__(self, detector: BaseDetector):
        self.detector = detector
        self.feature_pipeline = ImageFeaturePipeline()

    async def predict_from_upload(self, file: UploadFile) -> dict[str, Any]:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gecersiz dosya formati. Lutfen bir gorsel (jpg, png) yukleyin.",
            )

        file_bytes = await file.read()
        image = self.feature_pipeline.process_bytes(file_bytes)
        output = self.detector.predict(image)
        output["filename"] = file.filename
        return output

