import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ml import train as training_module
from ml.prepare_data import SPLITS, OutputRecoveryError, convert_one_label, prepare_dataset
from ml.train import train_model


def _write_pair(raw_dir, split, name="board"):
    images = raw_dir / split / "images"
    labels = raw_dir / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    (images / f"{name}.jpg").write_bytes(b"test image")
    (labels / f"{name}.txt").write_text("0 0 640 640 1\n", encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / "raw"
    for split in SPLITS:
        _write_pair(raw, split)
    return raw, tmp_path / "interim", tmp_path / "processed"


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _training_config(tmp_path):
    config = {
        "model": {"type": "local-initial-weights.pt", "imgsz": 320},
        "train": {"epochs": 2, "device": "cpu"},
        "paths": {
            "project_dir": str(tmp_path / "runs"),
            "experiment_name": "experiment",
            "output_model_dir": str(tmp_path / "exported"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, config


def test_train_publishes_actual_checkpoint_instead_of_guessed_run_path(tmp_path):
    config_path, config = _training_config(tmp_path)
    guessed = tmp_path / "runs" / "experiment" / "weights" / "best.pt"
    guessed.parent.mkdir(parents=True)
    guessed.write_bytes(b"old checkpoint")
    actual = tmp_path / "runs" / "detect" / "experiment2" / "weights" / "best.pt"
    target = Path(config["paths"]["output_model_dir"]) / "pcb_defect_best.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previously published checkpoint")
    calls = {}

    class FakeModel:
        def train(self, **kwargs):
            calls.update(kwargs)
            actual.parent.mkdir(parents=True)
            actual.write_bytes(b"new checkpoint")
            self.trainer = SimpleNamespace(best=actual)

    model = FakeModel()
    factories = []

    def model_factory(weights):
        factories.append(weights)
        return model

    assert train_model(str(config_path), model_factory=model_factory) is model
    target = Path(config["paths"]["output_model_dir"]) / "pcb_defect_best.pt"
    assert target.read_bytes() == b"new checkpoint"
    assert factories == ["local-initial-weights.pt"]
    assert calls["project"] == config["paths"]["project_dir"]
    assert calls["name"] == "experiment"
    assert calls["imgsz"] == 320
    assert calls["device"] == "cpu"


@pytest.mark.parametrize("checkpoint", [None, "missing.pt"])
def test_train_missing_best_checkpoint_fails_and_preserves_published_model(tmp_path, checkpoint):
    config_path, config = _training_config(tmp_path)
    target = Path(config["paths"]["output_model_dir"]) / "pcb_defect_best.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"published checkpoint")
    best = tmp_path / checkpoint if checkpoint else None
    model = SimpleNamespace(trainer=SimpleNamespace(best=best), train=lambda **kwargs: None)
    with pytest.raises(FileNotFoundError, match="best checkpoint"):
        train_model(str(config_path), model_factory=lambda weights: model)
    assert target.read_bytes() == b"published checkpoint"


def test_train_supports_checkpoint_already_at_publication_path(tmp_path):
    config_path, config = _training_config(tmp_path)
    target = Path(config["paths"]["output_model_dir"]) / "pcb_defect_best.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"checkpoint")
    model = SimpleNamespace(trainer=SimpleNamespace(best=target), train=lambda **kwargs: None)
    train_model(str(config_path), model_factory=lambda weights: model)
    assert target.read_bytes() == b"checkpoint"


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt, SystemExit])
def test_train_copy_failure_or_cancellation_preserves_old_model(tmp_path, monkeypatch, failure):
    config_path, config = _training_config(tmp_path)
    target = Path(config["paths"]["output_model_dir"]) / "pcb_defect_best.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"published checkpoint")
    source = tmp_path / "best.pt"
    source.write_bytes(b"new checkpoint")
    model = SimpleNamespace(trainer=SimpleNamespace(best=source), train=lambda **kwargs: None)

    def interrupted_copy(source_path, destination):
        Path(destination).write_bytes(b"incomplete checkpoint")
        raise failure("simulated copy failure")

    monkeypatch.setattr(training_module.shutil, "copy2", interrupted_copy)
    with pytest.raises(failure, match="simulated copy failure"):
        train_model(str(config_path), model_factory=lambda weights: model)
    assert target.read_bytes() == b"published checkpoint"
    assert list(target.parent.iterdir()) == [target]


def test_prepare_rebuilds_outputs_without_stale_pairs(dataset):
    raw, interim, processed = dataset
    _write_pair(raw, "train", "obsolete")
    first_stats = prepare_dataset(raw, interim, processed)
    assert first_stats["splits"]["train"]["num_images"] == 2
    (raw / "train" / "images" / "obsolete.jpg").unlink()
    (raw / "train" / "labels" / "obsolete.txt").unlink()

    stats = prepare_dataset(raw, interim, processed)
    assert not (interim / "train" / "labels" / "obsolete.txt").exists()
    assert not (processed / "train" / "images" / "obsolete.jpg").exists()
    assert not (processed / "train" / "labels" / "obsolete.txt").exists()
    assert stats["splits"]["train"]["num_images"] == 1
    assert stats["total_class_distribution"]["open"] == 3
    assert (processed / "train" / "labels" / "board.txt").read_text() == (
        "0 0.500000 0.500000 1.000000 1.000000"
    )
    assert json.loads((processed / "data_stats.json").read_text()) == stats
    expected = _snapshot(processed)
    prepare_dataset(raw, interim, processed)
    assert _snapshot(processed) == expected


@pytest.mark.parametrize("missing_kind, message", [("labels", "missing labels"), ("images", "missing images")])
def test_prepare_rejects_unpaired_raw_files_without_using_old_outputs(dataset, missing_kind, message):
    raw, interim, processed = dataset
    prepare_dataset(raw, interim, processed)
    before = _snapshot(interim), _snapshot(processed)
    extension = "txt" if missing_kind == "labels" else "jpg"
    (raw / "valid" / missing_kind / f"board.{extension}").unlink()
    with pytest.raises(ValueError, match=message):
        prepare_dataset(raw, interim, processed)
    assert (_snapshot(interim), _snapshot(processed)) == before
    assert not list(processed.parent.glob(".prepare-data-*"))


@pytest.mark.parametrize(
    "row, message",
    [
        ("0 0 20 20", "expected xmin"),
        ("zero 0 20 20 1", "must be integers"),
        ("0 0 20 20 7", "class_id must be between"),
        ("20 0 20 20 1", "positive area"),
        ("-1 0 20 20 1", "positive area"),
        ("0 0 641 20 1", "positive area"),
    ],
)
def test_prepare_rejects_invalid_late_split_without_replacing_outputs(dataset, row, message):
    raw, interim, processed = dataset
    prepare_dataset(raw, interim, processed)
    before = _snapshot(interim), _snapshot(processed)
    (raw / "test" / "labels" / "board.txt").write_text(row, encoding="utf-8")
    with pytest.raises(ValueError, match=message) as error:
        prepare_dataset(raw, interim, processed)
    assert "board.txt:1" in str(error.value)
    assert (_snapshot(interim), _snapshot(processed)) == before


def test_prepare_restores_previous_outputs_when_publication_fails(dataset, monkeypatch):
    raw, interim, processed = dataset
    prepare_dataset(raw, interim, processed)
    before = _snapshot(interim), _snapshot(processed)
    _write_pair(raw, "train", "new")
    original_replace = Path.replace

    def fail_new_processed_once(source, target):
        if source.name == "processed" and source.parent.name.startswith(".prepare-data-"):
            raise OSError("simulated publication error")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_new_processed_once)
    with pytest.raises(OSError, match="publication error"):
        prepare_dataset(raw, interim, processed)
    assert (_snapshot(interim), _snapshot(processed)) == before


def test_prepare_rejects_output_paths_that_overlap_raw_data(dataset):
    raw, _, _ = dataset
    before = _snapshot(raw)
    with pytest.raises(ValueError, match="overlap raw data"):
        prepare_dataset(raw, raw / "interim", raw / "processed")
    assert _snapshot(raw) == before


def test_prepare_preserves_backups_when_publication_and_rollback_fail(dataset, monkeypatch):
    raw, interim, processed = dataset
    prepare_dataset(raw, interim, processed)
    before = _snapshot(interim), _snapshot(processed)
    _write_pair(raw, "train", "new")
    original_replace = Path.replace

    def fail_publication_and_recovery(source, target):
        if source.name == "processed" and source.parent.name.startswith(".prepare-data-"):
            raise OSError("simulated publication error")
        if source.name == "previous-1":
            raise OSError("simulated rollback error")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_publication_and_recovery)
    with pytest.raises(OutputRecoveryError, match="backups preserved") as error:
        prepare_dataset(raw, interim, processed)
    workspace = error.value.workspace
    assert workspace.parent == processed.parent
    assert workspace.is_dir()
    assert (_snapshot(workspace / "previous-0"), _snapshot(workspace / "previous-1")) == before


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("moment", ["after_backup", "after_publish"])
def test_prepare_cancellation_restores_outputs_after_completed_rename(
    dataset, monkeypatch, interruption, moment
):
    raw, interim, processed = dataset
    prepare_dataset(raw, interim, processed)
    before = _snapshot(interim), _snapshot(processed)
    _write_pair(raw, "train", "new")
    original_replace = Path.replace

    def interrupt_after_rename(source, target):
        result = original_replace(source, target)
        backup_completed = target.name == "previous-0"
        publication_completed = (
            source.name == "interim" and source.parent.name.startswith(".prepare-data-")
        )
        if (moment == "after_backup" and backup_completed) or (
            moment == "after_publish" and publication_completed
        ):
            raise interruption("simulated cancellation after rename")
        return result

    monkeypatch.setattr(Path, "replace", interrupt_after_rename)
    with pytest.raises(interruption, match="simulated cancellation after rename"):
        prepare_dataset(raw, interim, processed)
    assert (_snapshot(interim), _snapshot(processed)) == before
    assert not list(processed.parent.glob(".prepare-data-*"))


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_prepare_preserves_backups_when_rollback_is_cancelled(dataset, monkeypatch, interruption):
    raw, interim, processed = dataset
    prepare_dataset(raw, interim, processed)
    before = _snapshot(interim), _snapshot(processed)
    _write_pair(raw, "train", "new")
    original_replace = Path.replace

    def cancel_recovery(source, target):
        if source.name == "processed" and source.parent.name.startswith(".prepare-data-"):
            raise OSError("simulated publication error")
        if source.name == "previous-1":
            raise interruption("simulated cancellation during rollback")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", cancel_recovery)
    with pytest.raises(OutputRecoveryError, match="backups preserved") as error:
        prepare_dataset(raw, interim, processed)
    workspace = error.value.workspace
    assert workspace.is_dir()
    assert (_snapshot(workspace / "previous-0"), _snapshot(workspace / "previous-1")) == before


def test_convert_allows_explicit_empty_labels_for_negative_images(tmp_path):
    source = tmp_path / "negative.txt"
    target = tmp_path / "converted.txt"
    source.write_text("\n", encoding="utf-8")
    assert convert_one_label(source, target) == 0
    assert target.read_text() == ""
