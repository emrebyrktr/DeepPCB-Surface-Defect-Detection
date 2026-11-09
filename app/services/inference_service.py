from typing import Any

from fastapi import HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.base_detector import BaseDetector
from app.core.config import settings
from app.services.feature_pipeline import ImageFeaturePipeline


class InferenceService:
    """Yüklemeyi doğrular ve tahmini thread pool'da çalıştırır."""

    def __init__(
        self,
        detector: BaseDetector,
        max_upload_bytes: int | None = None,
        max_image_pixels: int | None = None,
    ):
        self.detector = detector
        self.max_upload_bytes = settings.MAX_UPLOAD_BYTES if max_upload_bytes is None else max_upload_bytes
        if self.max_upload_bytes <= 0:
            raise ValueError("Dosya boyut limiti pozitif olmali.")
        self.feature_pipeline = ImageFeaturePipeline(
            max_image_pixels=settings.MAX_IMAGE_PIXELS if max_image_pixels is None else max_image_pixels,
        )

    def _predict_bytes(self, file_bytes: bytes) -> dict[str, Any]:
        image = self.feature_pipeline.process_bytes(file_bytes)
        return self.detector.predict(image)

    async def predict_from_upload(self, file: UploadFile) -> dict[str, Any]:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gecersiz dosya formati. Lutfen bir gorsel (jpg, png) yukleyin.",
            )

        file_bytes = await file.read(self.max_upload_bytes + 1)
        if len(file_bytes) > self.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Dosya boyutu izin verilen limiti asiyor.",
            )
        output = await run_in_threadpool(self._predict_bytes, file_bytes)
        output["filename"] = file.filename
        return output

