import cv2
import numpy as np
from fastapi import HTTPException, status


class ImageFeaturePipeline:
    """Görsel verisini model için ön hazırlık aşamalarından geçiren işlem boru hattı."""

    def __init__(self, max_image_pixels: int = 25_000_000):
        if max_image_pixels <= 0:
            raise ValueError("Gorsel piksel limiti pozitif olmali.")
        self.max_image_pixels = max_image_pixels

    def process_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Görsel bayt dizisini OpenCV matrisine dönüştürür ve doğrular."""
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bos gorsel verisi gonderildi.",
            )

        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gorsel verisi okunamadi veya bozuk.",
            )

        if image.shape[0] * image.shape[1] > self.max_image_pixels:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Gorselin piksel sayisi izin verilen limiti asiyor.",
            )

        return image
