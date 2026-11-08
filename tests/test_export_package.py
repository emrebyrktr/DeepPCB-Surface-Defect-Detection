import json
from types import SimpleNamespace

import pytest

from ml import export_package
from ml.tracker import file_sha256


@pytest.fixture
def export_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.pt"
    source.write_bytes(b"trained-checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  imgsz: 320\ntrain:\n  epochs: 5\n")
    preprocessing = tmp_path / "app/core/onnx_detector.py"
    preprocessing.parent.mkdir(parents=True)
    preprocessing.write_text("# preprocessing implementation\n")
    calls = {}

    def export(**kwargs):
        calls.update(kwargs)
        result = tmp_path / "export.onnx"
        result.write_bytes(b"exported-onnx")
        return str(result)

    monkeypatch.setattr(
        export_package, "_load_yolo",
        lambda path: SimpleNamespace(export=export, names={1: "short", 0: "open"}),
    )
    monkeypatch.setattr(
        export_package, "provenance",
        lambda: {"git_commit": "export-commit", "git_dirty": "true", "dvc_data_hash": "data.dir"},
    )
    return source, config, preprocessing, calls


def test_export_manifest_binds_files_config_and_preprocessing(export_inputs, tmp_path):
    source, config, preprocessing, calls = export_inputs
    output = tmp_path / "package"
    output.mkdir()
    (output / "metadata.json").write_text('{"metrics":{"test_map50":0.99}}')
    metadata = export_package.export_model(
        str(source), str(output), config_path=str(config), model_version="2.1.0",
        training_run_id="training-run",
    )
    assert calls == {"format": "onnx", "imgsz": 320}
    assert metadata["version"] == "2.1.0"
    assert metadata["classes"] == ["open", "short"]
    assert metadata["model_sha256"] == file_sha256(output / "pcb_defect_best.onnx")
    assert metadata["source_model_sha256"] == file_sha256(source)
    assert metadata["export_provenance"]["config_sha256"] == file_sha256(config)
    assert metadata["preprocessing"]["implementation_sha256"] == file_sha256(preprocessing)
    assert metadata["training_run_id"] == "training-run"
    assert "metrics" not in metadata
    assert json.loads((output / "metadata.json").read_text()) == metadata
    assert (output / "model.onnx").read_bytes() == (output / "pcb_defect_best.onnx").read_bytes()


def test_explicit_image_size_overrides_config(export_inputs, tmp_path):
    source, config, _, calls = export_inputs
    metadata = export_package.export_model(
        str(source), str(tmp_path / "package"), imgsz=640, config_path=str(config)
    )
    assert calls["imgsz"] == 640
    assert metadata["imgsz"] == 640
    assert metadata["version"] == "1.0.0"


@pytest.mark.parametrize("size", [0, -1, True, "640"])
def test_invalid_export_size_fails_before_export(export_inputs, tmp_path, size):
    source, config, _, calls = export_inputs
    with pytest.raises(ValueError, match="positive integer"):
        export_package.export_model(
            str(source), str(tmp_path / "package"), imgsz=size, config_path=str(config)
        )
    assert calls == {}


def test_empty_model_version_fails_before_export(export_inputs, tmp_path):
    source, config, _, calls = export_inputs
    with pytest.raises(ValueError, match="model_version"):
        export_package.export_model(
            str(source), str(tmp_path / "package"), config_path=str(config), model_version=" "
        )
    assert calls == {}
