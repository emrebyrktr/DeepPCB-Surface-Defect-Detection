from typing import Any


def filter_detections_by_confidence(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Belirli güven eşiğinin üzerindeki tespitleri filtreler."""
    return [d for d in detections if d.get("confidence", 0.0) >= threshold]

