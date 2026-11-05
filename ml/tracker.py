import os
import hashlib
import subprocess
import platform
import yaml
from typing import Dict, Any, Optional, List
import torch
import mlflow

class MLflowTracker:
    def __init__(
        self,
        experiment_name: str = "DeepPCB",
        tracking_uri: Optional[str] = None,
    ):
        if tracking_uri:
            tracking_uri = os.path.abspath(tracking_uri)
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        except Exception:
            return "unknown"

    def _get_dvc_hash(self, dvc_file: str = "data/processed.dvc") -> str:
        if os.path.exists(dvc_file):
            try:
                with open(dvc_file, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                return content.get("outs", [{}])[0].get("md5", "unknown")
            except Exception:
                return "unknown"
        return "not_found"

    def _get_file_sha256(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return "not_found"
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
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
                for path in artifacts:
                    if os.path.exists(path):
                        if os.path.isdir(path):
                            mlflow.log_artifacts(path, artifact_path=os.path.basename(path))
                        else:
                            mlflow.log_artifact(path)

            if model_path and os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="weights")