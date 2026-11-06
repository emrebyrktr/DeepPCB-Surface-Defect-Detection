from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """YAML dosyasını güvenle yükler."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Konfigurasyon dosyasi bulunamadi: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if data is not None else {}


def get_app_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """FastAPI servis ayarlarını döndürür."""
    return load_yaml(config_path).get("app", {})


def get_inference_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Çıkarım (inference) ayarlarını döndürür."""
    return load_yaml(config_path).get("inference", {})
