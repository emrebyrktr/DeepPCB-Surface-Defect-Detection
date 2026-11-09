"""Model, data and code provenance with optional MLflow tracking."""

import hashlib
import importlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


def file_sha256(filepath: str | Path) -> str:
    path = Path(filepath)
    if not path.is_file():
        return "not_found"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, str]:
    """Return the commit ID and whether the working tree has changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return {"git_commit": commit, "git_dirty": str(bool(status.strip())).lower()}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def dvc_data_hash(
    lock_file: str | Path = "dvc.lock",
    dvc_file: str | Path = "data/processed.dvc",
    data_path: str = "data/processed",
) -> str:
    """Read the processed-data hash from the lock file or a legacy DVC pointer."""
    try:
        with Path(lock_file).open(encoding="utf-8") as stream:
            locked = yaml.safe_load(stream) or {}
        for stage in locked.get("stages", {}).values():
            for output in stage.get("outs", []):
                output_path = str(output.get("path", "")).replace("\\", "/")
                if output_path.removeprefix("./").rstrip("/") == data_path.rstrip("/"):
                    algorithm = output.get("hash", "md5")
                    return str(output.get(algorithm, "unknown"))
    except (OSError, yaml.YAMLError, AttributeError, TypeError):
        pass
    try:
        with Path(dvc_file).open(encoding="utf-8") as stream:
            pointer = yaml.safe_load(stream) or {}
        output = pointer.get("outs", [{}])[0]
        return str(output.get(output.get("hash", "md5"), "unknown"))
    except FileNotFoundError:
        return "not_found"
    except (OSError, yaml.YAMLError, AttributeError, IndexError, TypeError):
        return "unknown"


def provenance() -> dict[str, str]:
    return {**git_state(), "dvc_data_hash": dvc_data_hash()}


class MLflowTracker:
    def __init__(
        self,
        experiment_name: str = "DeepPCB",
        tracking_uri: str | None = None,
        client: Any | None = None,
    ):
        # Load MLflow only when creating a tracker.
        self.mlflow = client if client is not None else importlib.import_module("mlflow")
        if tracking_uri:
            uri = str(tracking_uri)
            if not uri.startswith(("http://", "https://", "file://", "sqlite://")):
                uri = Path(tracking_uri).resolve().as_uri()
            self.mlflow.set_tracking_uri(uri)
        self.mlflow.set_experiment(experiment_name)

    def _get_git_commit(self) -> str:
        return git_state()["git_commit"]

    def _get_dvc_hash(self, dvc_file: str = "data/processed.dvc") -> str:
        return dvc_data_hash(dvc_file=dvc_file)

    def _get_file_sha256(self, filepath: str) -> str:
        return file_sha256(filepath)

    def log_experiment(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[str] | None = None,
        model_path: str | None = None,
        extra_tags: dict[str, str] | None = None,
        actual_device: str | None = None,
    ) -> str:
        try:
            torch_version = version("torch")
        except PackageNotFoundError:
            torch_version = "not_installed"
        tags = {
            **provenance(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch_version,
            # Record the device selected for this run.
            "device": str(actual_device or params.get("device") or "unknown"),
        }
        if model_path:
            tags["model_sha256"] = file_sha256(model_path)
        if extra_tags:
            tags.update(extra_tags)

        with self.mlflow.start_run(run_name=run_name) as run:
            self.mlflow.set_tags(tags)
            self.mlflow.log_params(params)
            self.mlflow.log_metrics(metrics)
            for item in artifacts or []:
                path = Path(item)
                if path.is_dir():
                    self.mlflow.log_artifacts(str(path), artifact_path=path.name)
                elif path.is_file():
                    self.mlflow.log_artifact(str(path))
            if model_path and Path(model_path).is_file():
                self.mlflow.log_artifact(str(Path(model_path)), artifact_path="weights")
            return str(run.info.run_id)

    def log_report(self, run_id: str, report_path: str | Path) -> None:
        """Attach an evaluation report to an existing run."""
        self.mlflow.tracking.MlflowClient().log_artifact(run_id, str(report_path))
