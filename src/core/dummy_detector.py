from typing import Any, Dict, List, Tuple
import numpy as np

from src.core.base_detector import BaseDetector


class DummyDetector(BaseDetector):
    """Testler ve prototipleme icin sahte cikarim motoru."""

    def __init__(self, classes: List[str] = None):
        self.classes = classes or ["open", "short", "mousebite", "spur", "copper", "pin-hole"]

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        return np.zeros((1, 3, 640, 640), dtype=np.float32), 1.0, 0.0, 0.0

    def forward(self, blob: np.ndarray) -> np.ndarray:
        return np.zeros((1, 10, 8400), dtype=np.float32)

    def postprocess(
        self,
        outputs: Any,
        orig_shape: Tuple[int, int],
        scale_info: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> List[Dict[str, Any]]:
        return [
            {
                "box": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
                "confidence": 0.95,
                "class_id": 0,
                "class_name": self.classes[0],
            }
        ]

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        return {
            "detections": [
                {
                    "box": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
                    "confidence": 0.95,
                    "class_id": 0,
                    "class_name": self.classes[0],
                }
            ],
            "total_defects": 1,
            "latency_ms": {
                "preprocess": 1.2,
                "inference": 5.4,
                "postprocess": 0.8,
                "total": 7.4,
            },
        }

