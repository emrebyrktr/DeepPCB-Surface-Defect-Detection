import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ml import evaluate, tracker


def test_dvc_hash_uses_processed_output_not_raw_dependency(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/processed.dvc").write_text("outs:\n- md5: legacy.dir\n")
    (tmp_path / "dvc.lock").write_text(
        "stages:\n  prepare_data:\n    deps:\n    - path: data/raw\n"
        "      md5: raw.dir\n    outs:\n    - path: data/processed\n"
        "      hash: md5\n      md5: processed.dir\n"
    )
    assert tracker.dvc_data_hash() == "processed.dir"


def test_dvc_hash_supports_legacy_pointer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pointer = tmp_path / "legacy.dvc"
    pointer.write_text("outs:\n- md5: legacy.dir\n")
    assert tracker.dvc_data_hash(dvc_file=pointer) == "legacy.dir"
    assert tracker.dvc_data_hash() == "not_found"


def test_git_state_marks_uncommitted_work(monkeypatch):
    responses = iter([b"commit-id\n", b" M ml/train.py\n"])
    monkeypatch.setattr(tracker.subprocess, "check_output", lambda *a, **k: next(responses))
    assert tracker.git_state() == {"git_commit": "commit-id", "git_dirty": "true"}


def test_git_state_does_not_invent_commit_outside_repository(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(tracker.subprocess, "check_output", unavailable)
    assert tracker.git_state() == {"git_commit": "unknown", "git_dirty": "unknown"}


class FakeMLflow:
    def __init__(self):
        self.tags = {}
        self.artifacts = []
        self.tracking = SimpleNamespace(MlflowClient=lambda: self)

    def set_tracking_uri(self, uri):
        self.uri = uri

    def set_experiment(self, name):
        self.experiment = name

    @contextmanager
    def start_run(self, **kwargs):
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    def set_tags(self, tags):
        self.tags = tags

    def log_params(self, params):
        self.params = params

    def log_metrics(self, metrics):
        self.metrics = metrics

    def log_artifact(self, *args, **kwargs):
        self.artifacts.append((args, kwargs))

    def log_artifacts(self, *args, **kwargs):
        self.artifacts.append((args, kwargs))


def test_tracker_returns_run_id_and_records_requested_runtime(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(tracker, "provenance", lambda: {"dvc_data_hash": "processed.dir"})
    fake = FakeMLflow()
    tracking = tracker.MLflowTracker(client=fake, tracking_uri=str(tmp_path / "mlruns"))
    run_id = tracking.log_experiment(
        "cpu-eval", {"device": "0"}, {"recall": 0.9},
        model_path=str(model), actual_device="cpu",
    )
    assert run_id == "run-123"
    assert fake.tags["device"] == "cpu"
    assert fake.tags["model_sha256"] == tracker.file_sha256(model)
    assert fake.tags["dvc_data_hash"] == "processed.dir"
    assert fake.uri == (tmp_path / "mlruns").as_uri()
    tracking.log_report(run_id, tmp_path / "metrics.json")
    assert fake.artifacts[-1][0] == (run_id, str(tmp_path / "metrics.json"))


def evaluation_metrics(tmp_path):
    return SimpleNamespace(
        box=SimpleNamespace(
            mp=0.9, mr=0.8, map50=0.85, map=0.7,
            ap_class_index=np.array([4]), ap50=np.array([0.8]),
            ap=np.array([[0.7, 0.5]]), p=np.array([0.9]), r=np.array([0.6]),
        ),
        names={0: "open", 4: "copper"},
        speed={"preprocess": 1, "inference": 2, "postprocess": 3},
        save_dir=tmp_path / "figures",
    )


def test_class_metrics_handle_classes_missing_from_split(tmp_path):
    result = evaluate._metrics_dict(evaluation_metrics(tmp_path), "test")
    assert result["test_map50_copper"] == 0.8
    assert result["test_map50_95_copper"] == pytest.approx(0.6)
    assert result["test_recall_copper"] == 0.6
    assert "test_map50_open" not in result


def test_nonfinite_metrics_are_rejected(tmp_path):
    metrics = evaluation_metrics(tmp_path)
    metrics.box.mr = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        evaluate._metrics_dict(metrics, "test")


def test_macro_f1_is_mean_of_class_f1_not_f1_of_mean_precision_recall(tmp_path):
    metrics = evaluation_metrics(tmp_path)
    metrics.box.mp = metrics.box.mr = 0.5
    metrics.box.ap_class_index = np.array([0, 4])
    metrics.box.f1 = np.array([0.18, 0.18])
    metrics.box.p = np.array([0.1, 0.9])
    metrics.box.r = np.array([0.9, 0.1])

    result = evaluate._metrics_dict(metrics, "test")

    assert result["test_f1"] == pytest.approx(0.18)
    assert result["test_f1_open"] == pytest.approx(0.18)
    assert result["test_f1_copper"] == pytest.approx(0.18)


def test_evaluation_resolves_project_dataset_path_without_changing_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/processed").mkdir(parents=True)
    source = tmp_path / "configs/data.yaml"
    source.parent.mkdir()
    original = "path: data/processed\ntest: test/images\nnames: [open]\n"
    source.write_text(original, encoding="utf-8")

    resolved = evaluate._resolved_data_config(str(source), tmp_path / "reports/metrics.json")

    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert Path(data["path"]) == tmp_path / "data/processed"
    assert source.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("split", ["test", "val"])
def test_evaluation_writes_report_without_rewriting_model_metadata(tmp_path, monkeypatch, split):
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"candidate")
    metadata = tmp_path / "metadata.json"
    original_metadata = '{"model_sha256":"another-model","version":"previous"}'
    metadata.write_text(original_metadata)
    config = tmp_path / "config.yaml"
    config.write_text("train:\n  device: '0'\ninference:\n  device: cpu\n")
    data = tmp_path / "data.yaml"
    data.write_text("names: [open, copper]\n")
    calls = {}

    def validate(**kwargs):
        calls["validation"] = kwargs
        return evaluation_metrics(tmp_path)

    class FakeTracker:
        def __init__(self, **kwargs):
            pass

        def log_experiment(self, **kwargs):
            calls["tracking"] = kwargs
            return "evaluation-run"

        def log_report(self, run_id, report_path):
            calls["report"] = (run_id, json.loads(report_path.read_text()))

    monkeypatch.setattr(evaluate, "_load_yolo", lambda path: SimpleNamespace(val=validate))
    monkeypatch.setattr(evaluate, "MLflowTracker", FakeTracker)
    monkeypatch.setattr(evaluate, "provenance", lambda: {"git_commit": "current"})
    output = tmp_path / "reports/metrics.json"
    result = evaluate.evaluate_model(
        str(model), str(data), str(config), split=split, output_path=str(output)
    )
    report = json.loads(output.read_text())
    assert calls["validation"]["split"] == split
    assert calls["validation"]["device"] == "cpu"
    assert calls["validation"]["plots"] is True
    assert calls["validation"]["conf"] == 0.001
    assert calls["validation"]["iou"] == 0.7
    assert calls["validation"]["workers"] == 0
    assert calls["tracking"]["actual_device"] == "cpu"
    assert str(metadata) not in calls["tracking"]["artifacts"]
    assert report["model_sha256"] == tracker.file_sha256(model)
    assert report["metrics"] == result
    assert report["mlflow_run_id"] == "evaluation-run"
    assert calls["report"] == ("evaluation-run", report)
    assert metadata.read_text() == original_metadata


def test_evaluation_rejects_training_split_before_loading_model():
    with pytest.raises(ValueError, match="split"):
        evaluate.evaluate_model("unused.onnx", split="train")
