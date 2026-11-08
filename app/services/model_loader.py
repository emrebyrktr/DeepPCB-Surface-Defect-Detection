from pathlib import Path

from app.core.base_detector import BaseDetector
from app.core.dummy_detector import DummyDetector
from app.core.onnx_detector import ONNXDetector
from app.utils.logger import setup_logger

logger = setup_logger("model_loader")


class ModelLoader:
    """Model yükleme servisi."""

    @staticmethod
    def load_detector(
        model_path: str = "models/pcb_defect_best.onnx",
        config_path: str = "configs/config.yaml",
        fallback_dummy: bool = False,
        *,
        metadata_path: str | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> BaseDetector:
        path = Path(model_path)
        if path.exists():
            logger.info(f"ONNX modeli yukleniyor: {path}")
            return ONNXDetector(
                config_path=config_path,
                model_path=str(path),
                metadata_path=metadata_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

        if fallback_dummy:
            logger.warning(f"Model ({path}) bulunamadi. DummyDetector ile baslatiliyor.")
            return DummyDetector()

        raise FileNotFoundError(f"Model dosyasi bulunamadi: {path}")

