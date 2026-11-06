import hashlib
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import torch
import mlflow

class MLflowTracker:
    def __init__(
        self,
        experiment_name: str = "DeepPCB",
        tracking_uri: Optional[str] = None,
    ):
        if tracking_uri:
            uri_str = str(tracking_uri)
            if uri_str.startswith(("http://", "https://", "file://", "sqlite://")):
                resolved_uri = uri_str
            else:
                resolved_uri = Path(tracking_uri).resolve().as_uri()
            mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment(experiment_name)

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        except Exception:
            return "unknown"

    def _get_dvc_hash(self, dvc_file: str = "data/processed.dvc") -> str:
        dvc_path = Path(dvc_file)
        if dvc_path.exists():
            try:
                with open(dvc_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                return content.get("outs", [{}])[0].get("md5", "unknown")
            except Exception:
                return "unknown"
        return "not_found"

    def _get_file_sha256(self, filepath: str) -> str:
        file_path = Path(filepath)
        if not file_path.exists():
            return "not_found"
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def log_experiment(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        extra_tags: Optional[Dict[str, str]] = None,
    ):
        tags = {
            "dvc_data_hash": self._get_dvc_hash(),
            "git_commit": self._get_git_commit(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        if model_path:
            tags["model_sha256"] = self._get_file_sha256(model_path)

        if extra_tags:
            tags.update(extra_tags)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(tags)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            if artifacts:
                for item in artifacts:
                    artifact_path = Path(item)
                    if artifact_path.exists():
                        if artifact_path.is_dir():
                            mlflow.log_artifacts(str(artifact_path), artifact_path=artifact_path.name)
                        else:
                            mlflow.log_artifact(str(artifact_path))

            if model_path and Path(model_path).exists():
                mlflow.log_artifact(str(Path(model_path)), artifact_path="weights")
