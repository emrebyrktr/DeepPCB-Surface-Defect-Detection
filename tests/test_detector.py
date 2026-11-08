import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf

from app.api.main import app
from app.core import onnx_detector as detector_module
from app.core.config import Settings
from app.core.onnx_detector import ONNXDetector
from app.services.model_loader import ModelLoader

api_main = importlib.import_module("app.api.main")


@pytest.fixture
def model_files(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"not an ONNX model")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "model_name": "fixture-model",
        "version": "2.3.4",
        "classes": ["open", "short"],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("inference:\n  conf_threshold: 0.3\n  iou_threshold: 0.45\n", encoding="utf-8")
    return model_path, metadata_path, config_path


@pytest.fixture
def runtime_stub(monkeypatch):
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

        def get_outputs(self):
            return [SimpleNamespace(name="output0", shape=[1, 6, 8400])]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(detector_module.ort, "InferenceSession", lambda *args, **kwargs: Session())


def test_model_loader_requires_artifact_by_default(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model dosyasi"):
        ModelLoader.load_detector(model_path=str(tmp_path / "missing.onnx"))


def test_lifespan_rejects_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main.settings, "MODEL_PATH", str(tmp_path / "missing.onnx"))
    with pytest.raises(FileNotFoundError, match="Model dosyasi"):
        with TestClient(app):
            pytest.fail("Model eksikken uygulama acilmamali.")


def test_lifespan_rejects_invalid_model(monkeypatch, model_files):
    model_path, metadata_path, config_path = model_files
    monkeypatch.setattr(api_main.settings, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(api_main.settings, "METADATA_PATH", str(metadata_path))
    monkeypatch.setattr(api_main.settings, "config_path", config_path)
    with pytest.raises(InvalidProtobuf):
        with TestClient(app):
            pytest.fail("Gecersiz ONNX dosyasiyla uygulama acilmamali.")


def test_lifespan_forwards_environment_settings(monkeypatch, model_files, runtime_stub):
    model_path, metadata_path, config_path = model_files
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("CONF_THRESHOLD", "0.9")
    monkeypatch.setenv("IOU_THRESHOLD", "0.8")
    monkeypatch.setenv("DEVICE", "cuda")
    settings = Settings(config_path=config_path)
    monkeypatch.setattr(api_main, "settings", settings)

    with TestClient(app) as client:
        detector = app.state.detector
        assert detector.model_path == model_path
        assert detector.metadata_path == metadata_path
        assert detector.conf_threshold == 0.9
        assert detector.iou_threshold == 0.8
        health = client.get("/api/v1/health").json()
        assert health["model_name"] == "fixture-model"
        assert health["model_version"] == "2.3.4"
        assert health["device"] == "cpu"
    assert app.state.detector is None


def test_zero_thresholds_are_respected(model_files, runtime_stub):
    model_path, metadata_path, config_path = model_files
    detector = ONNXDetector(
        model_path=str(model_path), metadata_path=str(metadata_path),
        config_path=str(config_path), conf_threshold=0.0, iou_threshold=0.0,
    )
    assert detector.conf_threshold == 0.0
    assert detector.iou_threshold == 0.0


@pytest.mark.parametrize("field,value", [("conf_threshold", -0.1), ("iou_threshold", 1.1)])
def test_invalid_thresholds_fail_before_loading(model_files, field, value):
    model_path, metadata_path, config_path = model_files
    with pytest.raises(ValueError, match="esikleri"):
        ONNXDetector(
            model_path=str(model_path), metadata_path=str(metadata_path),
            config_path=str(config_path), **{field: value},
        )


def test_metadata_is_required(model_files, runtime_stub):
    model_path, metadata_path, config_path = model_files
    metadata_path.unlink()
    with pytest.raises(FileNotFoundError):
        ONNXDetector(
            model_path=str(model_path), metadata_path=str(metadata_path), config_path=str(config_path),
        )


@pytest.mark.parametrize("field,value", [("classes", []), ("model_name", ""), ("version", None)])
def test_invalid_metadata_rejected(model_files, runtime_stub, field, value):
    model_path, metadata_path, config_path = model_files
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        ONNXDetector(
            model_path=str(model_path), metadata_path=str(metadata_path), config_path=str(config_path),
        )


def test_model_hash_mismatch_fails_before_runtime(monkeypatch, model_files):
    model_path, metadata_path, config_path = model_files
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def unexpected_runtime(*args, **kwargs):
        pytest.fail("Hash eslesmeden native ONNX oturumu acilmamali.")

    monkeypatch.setattr(detector_module.ort, "InferenceSession", unexpected_runtime)
    with pytest.raises(ValueError, match="SHA256.*eslesmiyor"):
        ONNXDetector(
            model_path=str(model_path), metadata_path=str(metadata_path), config_path=str(config_path),
        )


def test_matching_model_hash_is_accepted(model_files, runtime_stub):
    model_path, metadata_path, config_path = model_files
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    detector = ONNXDetector(
        model_path=str(model_path), metadata_path=str(metadata_path), config_path=str(config_path),
    )
    assert detector.model_name == "fixture-model"


def test_nms_preserves_overlapping_different_classes():
    detector = ONNXDetector.__new__(ONNXDetector)
    detector.classes = ["open", "short"]
    detector.conf_threshold = 0.3
    detector.iou_threshold = 0.45
    outputs = np.zeros((1, 6, 3), dtype=np.float32)
    outputs[0, :4, :] = np.array([[100], [100], [50], [50]])
    outputs[0, 4, 0] = 0.95
    outputs[0, 4, 1] = 0.70  # Ayni siniftaki tekrar bastirilmali.
    outputs[0, 5, 2] = 0.90  # Farkli siniftaki ortusen kusur korunmali.

    result = detector.postprocess(outputs, (640, 640), (1.0, 0.0, 0.0))

    assert [detection["class_name"] for detection in result] == ["open", "short"]
    assert [detection["confidence"] for detection in result] == [0.95, 0.9]


@pytest.mark.integration
def test_real_onnx_lifespan_and_prediction():
    if os.getenv("RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("Gercek model testi icin RUN_MODEL_INTEGRATION=1 ayarlayin.")
    if not Path(api_main.settings.MODEL_PATH).is_file() or not Path(api_main.settings.METADATA_PATH).is_file():
        pytest.skip("Gercek ONNX model paketi mevcut degil.")
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["model_name"] != "DummyDetector"
        assert health.json()["device"] == "cpu"
        _, encoded = cv2.imencode(".png", np.zeros((64, 64, 3), dtype=np.uint8))
        response = client.post(
            "/api/v1/predict", files={"file": ("integration.png", encoded.tobytes(), "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert isinstance(response.json()["detections"], list)
