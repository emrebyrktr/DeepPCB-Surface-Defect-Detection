from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: str | Path, *, required: bool = True) -> dict[str, Any]:
    """YAML ayar dosyasını sözlük olarak okur."""
    path = Path(file_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        if not required:
            return {}
        raise FileNotFoundError(f"Konfigurasyon dosyasi bulunamadi: {path}") from None

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Konfigurasyon bir YAML nesnesi olmali: {path}")
    return data


def get_app_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """FastAPI servis ayarlarını döndürür."""
    return load_yaml(config_path).get("app", {})


def get_inference_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Çıkarım (inference) ayarlarını döndürür."""
    return load_yaml(config_path).get("inference", {})
