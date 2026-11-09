import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from ml.tracker import MLflowTracker, file_sha256, provenance


def _load_yolo(model_path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path)


def _metrics_dict(metrics: Any, split: str) -> dict[str, float]:
    box = metrics.box
    results = {
        f"{split}_precision": float(box.mp),
        f"{split}_recall": float(box.mr),
        f"{split}_map50": float(box.map50),
        f"{split}_map50_95": float(box.map),
    }
    f1_values = getattr(box, "f1", [])
    if len(f1_values):
        results[f"{split}_f1"] = sum(float(value) for value in f1_values) / len(f1_values)
    for stage in ("preprocess", "inference", "postprocess"):
        results[f"speed_{stage}_ms"] = float(metrics.speed.get(stage, 0.0))
    results["speed_total_ms"] = sum(
        results[f"speed_{stage}_ms"] for stage in ("preprocess", "inference", "postprocess")
    )
    # AP rows refer to classes present in this split, not necessarily class IDs 0..N.
    ap50 = getattr(box, "ap50", [])
    class_ids = getattr(box, "ap_class_index", range(len(ap50)))
    for row, class_id in enumerate(class_ids):
        class_id = int(class_id)
        names = metrics.names
        if isinstance(names, dict):
            name = names.get(class_id, names.get(str(class_id), f"class_{class_id}"))
        else:
            name = names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
        for key, values in (
            ("map50", ap50),
            ("map50_95", getattr(box, "ap", [])),
            ("precision", getattr(box, "p", [])),
            ("recall", getattr(box, "r", [])),
            ("f1", f1_values),
        ):
            if row < len(values):
                value = values[row]
                if hasattr(value, "mean"):
                    value = value.mean()
                results[f"{split}_{key}_{name}"] = float(value)
        for key, values in (
            ("instances", getattr(metrics, "nt_per_class", None)),
            ("images", getattr(metrics, "nt_per_image", None)),
        ):
            if values is not None and class_id < len(values):
                results[f"{split}_{key}_{name}"] = float(values[class_id])
    if not all(math.isfinite(value) for value in results.values()):
        raise ValueError("Evaluation produced non-finite metrics")
    return results


def _resolved_data_config(data_yaml: str, output: Path) -> Path:
    source = Path(data_yaml).resolve()
    with source.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Dataset configuration must be a YAML mapping")
    root = Path(data.get("path") or source.parent)
    if not root.is_absolute():
        relative_to_cwd = Path.cwd() / root
        root = relative_to_cwd if relative_to_cwd.is_dir() else source.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    data["path"] = str(root)
    resolved = output.with_suffix(".data.yaml")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    return resolved


def evaluate_model(
    model_path: str,
    data_yaml: str = "configs/data.yaml",
    config_path: str = "configs/config.yaml",
    tracking_uri: str | None = None,
    split: str = "test",
    output_path: str = "reports/metrics.json",
    device: str | None = None,
) -> dict[str, float]:
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    # Use the configured inference device unless evaluation overrides it.
    selected_device = str(device or config.get("inference", {}).get("device", "cpu"))
    output = Path(output_path)
    resolved_data = _resolved_data_config(data_yaml, output)
    validation = {
        "data": str(resolved_data), "split": split, "device": selected_device,
        "imgsz": int(config.get("model", {}).get("imgsz", 640)),
        "batch": 1, "workers": 0, "conf": 0.001, "iou": 0.7, "plots": True,
        "project": str(output.parent / "evaluation"), "name": split,
    }
    model = _load_yolo(model_path)
    evaluated = model.val(**validation)
    results = _metrics_dict(evaluated, split)
    model_file = Path(model_path)
    artifacts = [str(evaluated.save_dir), str(config_path), str(data_yaml), str(resolved_data)]
    metadata_file = model_file.parent / "metadata.json"
    if metadata_file.is_file():
        with metadata_file.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("model_sha256") == file_sha256(model_file):
            artifacts.append(str(metadata_file))

    tracker = MLflowTracker(experiment_name="DeepPCB_Evaluation", tracking_uri=tracking_uri)
    run_id = tracker.log_experiment(
        run_name=f"{split}_evaluation",
        params={
            "data": data_yaml,
            "split": split,
            "device": selected_device,
            "config_sha256": file_sha256(config_path),
            "imgsz": validation["imgsz"],
            "confidence": validation["conf"],
            "nms_iou": validation["iou"],
        },
        metrics=results,
        artifacts=artifacts,
        model_path=model_path,
        extra_tags={"evaluation_split": split},
        actual_device=selected_device,
    )
    artifact_dir = Path(evaluated.save_dir)
    try:
        artifact_dir = artifact_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        pass
    report = {
        "schema_version": 1,
        "model_path": str(model_file),
        "model_sha256": file_sha256(model_file),
        "data_config_sha256": file_sha256(data_yaml),
        "config_sha256": file_sha256(config_path),
        "resolved_data_config_sha256": file_sha256(resolved_data),
        "evaluation_provenance": provenance(),
        "split": split,
        "device": selected_device,
        "mlflow_run_id": run_id,
        "metrics": results,
        "validation": validation,
        "artifacts_dir": artifact_dir.as_posix(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    tracker.log_report(run_id, output)
    print(results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pcb_defect_best.pt")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", default="reports/metrics.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        config_path=args.config,
        tracking_uri=args.tracking_uri,
        split=args.split,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
