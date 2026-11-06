import time
from typing import Any

from fastapi import UploadFile

from app.core.base_detector import BaseDetector
from app.services.feature_pipeline import ImageFeaturePipeline
from app.utils.validators import validate_image_file


class InferenceService:
    """Çıkarım iş akışını yöneten servis katmanı."""

    def __init__(self, detector: BaseDetector):
        self.detector = detector
        self.feature_pipeline = ImageFeaturePipeline()

    async def predict_from_upload(self, file: UploadFile) -> dict[str, Any]:
        """Yüklenen görsel dosyasını doğrular, çözer ve model çıkarımı gerçekleştirir."""
        validate_image_file(file)

        start_io = time.perf_counter()
        file_bytes = await file.read()
        io_time_ms = (time.perf_counter() - start_io) * 1000.0

        image = self.feature_pipeline.process_bytes(file_bytes)
        output = self.detector.predict(image)

        # İsteğe bağlı ek metrikler
        output["io_read_ms"] = round(io_time_ms, 2)
        output["filename"] = file.filename

        return output
