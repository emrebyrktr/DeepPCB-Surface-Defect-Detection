import os
from pathlib import Path
from typing import Any

from app.utils.config_loader import load_yaml


class Settings:
    """YAML ayarlarını okur; ortam değişkenlerini öncelikli kullanır."""

    def __init__(self, config_path: str | Path = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.data: dict[str, Any] = load_yaml(self.config_path, required=False)

        app_cfg = self.data.get("app", {})
        self.PROJECT_NAME: str = os.getenv("PROJECT_NAME", app_cfg.get("title", "Industrial PCB Defect Detection API"))
        self.VERSION: str = os.getenv("VERSION", app_cfg.get("version", "1.0.0"))
        self.API_V1_STR: str = "/api/v1"
        self.HOST: str = os.getenv("HOST", app_cfg.get("host", "0.0.0.0"))
        self.PORT: int = int(os.getenv("PORT", app_cfg.get("port", 8000)))

        infer_cfg = self.data.get("inference", {})
        self.MODEL_PATH: str = os.getenv("MODEL_PATH", infer_cfg.get("model_path", "models/pcb_defect_best.onnx"))
        self.METADATA_PATH: str = os.getenv("METADATA_PATH", infer_cfg.get("metadata_path", "models/metadata.json"))
        self.CONF_THRESHOLD: float = float(os.getenv("CONF_THRESHOLD", infer_cfg.get("conf_threshold", 0.30)))
        self.IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", infer_cfg.get("iou_threshold", 0.45)))
        self.DEVICE: str = os.getenv("DEVICE", infer_cfg.get("device", "cpu"))
        self.MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", infer_cfg.get("max_upload_bytes", 10 * 1024 * 1024)))
        self.MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", infer_cfg.get("max_image_pixels", 25_000_000)))
        if self.MAX_UPLOAD_BYTES <= 0 or self.MAX_IMAGE_PIXELS <= 0:
            raise ValueError("Gorsel boyut limitleri pozitif olmali.")

        self.CORS_ORIGINS: list[str] = ["*"]

settings = Settings()

