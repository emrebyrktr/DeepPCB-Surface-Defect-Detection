import cv2
import numpy as np
from fastapi import HTTPException, status


class ImageFeaturePipeline:
    """Görsel verisini model için ön hazırlık aşamalarından geçiren işlem boru hattı."""

    def __init__(self, target_size: int = 640):
        self.target_size = target_size

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

        return image

    def get_image_info(self, image: np.ndarray) -> dict[str, int]:
        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1
        return {"height": h, "width": w, "channels": c}

