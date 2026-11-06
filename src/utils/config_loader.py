from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """YAML dosyasini guvenle yukler."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Konfigurasyon dosyasi bulunamadi: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if data is not None else {}


def get_app_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """FastAPI servis ayarlarini dondurur."""
    return load_yaml(config_path).get("app", {})


def get_inference_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """Cikarim (inference) ayarlarini dondurur."""
    return load_yaml(config_path).get("inference", {})

