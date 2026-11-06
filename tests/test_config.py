from pathlib import Path
import json
import yaml
import pytest

from src.utils import load_yaml, get_app_config, get_inference_config


def test_config_yaml_exists():
    path = Path("configs/config.yaml")
    assert path.exists(), "configs/config.yaml dosyasi mevcut olmali"


def test_config_yaml_sections():
    cfg = load_yaml("configs/config.yaml")
    assert "app" in cfg
    assert "train" in cfg
    assert "inference" in cfg
    assert "paths" in cfg


def test_app_config():
    app_cfg = get_app_config("configs/config.yaml")
    assert "title" in app_cfg
    assert "version" in app_cfg
    assert app_cfg["version"] == "1.0.0"
    assert app_cfg.get("port") == 8000


def test_inference_config_thresholds():
    infer_cfg = get_inference_config("configs/config.yaml")
    conf = infer_cfg.get("conf_threshold")
    iou = infer_cfg.get("iou_threshold")

    assert isinstance(conf, (int, float))
    assert 0.0 < conf < 1.0
    assert isinstance(iou, (int, float))
    assert 0.0 < iou < 1.0


def test_data_yaml_structure():
    data_cfg = load_yaml("configs/data.yaml")
    assert data_cfg.get("nc") == 6
    names = data_cfg.get("names", [])
    assert len(names) == 6
    expected_classes = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
    assert names == expected_classes


def test_metadata_consistency():
    meta_path = Path("models/metadata.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta.get("version") == "1.0.0"
        data_cfg = load_yaml("configs/data.yaml")
        assert meta.get("classes") == data_cfg.get("names")

