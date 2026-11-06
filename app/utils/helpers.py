from typing import Any

import cv2
import numpy as np


def decode_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    """Byte dizisinden OpenCV BGR görsel matrisi üretir."""
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def format_latency(preprocess_ms: float, inference_ms: float, postprocess_ms: float) -> dict[str, float]:
    """Gecikme değerlerini milisaniye ve 2 basamak yuvarlama ile biçimlendirir."""
    total = preprocess_ms + inference_ms + postprocess_ms
    return {
        "preprocess": round(preprocess_ms, 2),
        "inference": round(inference_ms, 2),
        "postprocess": round(postprocess_ms, 2),
        "total": round(total, 2),
    }


def filter_detections_by_confidence(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Belirli güven eşiğinin üzerindeki tespitleri filtreler."""
    return [d for d in detections if d.get("confidence", 0.0) >= threshold]
