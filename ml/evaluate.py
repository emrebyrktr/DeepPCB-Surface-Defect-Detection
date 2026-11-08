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
        ):
            if row < len(values):
                value = values[row]
                if hasattr(value, "mean"):
                    value = value.mean()
                results[f"{split}_{key}_{name}"] = float(value)
    if not all(math.isfinite(value) for value in results.values()):
        raise ValueError("Evaluation produced non-finite metrics")
    return results


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
    # Use the production CPU target by default, independently of training hardware.
    selected_device = str(device or config.get("inference", {}).get("device", "cpu"))
    model = _load_yolo(model_path)
    evaluated = model.val(data=data_yaml, split=split, name=split, device=selected_device)
    results = _metrics_dict(evaluated, split)
    model_file = Path(model_path)
    artifacts = [str(evaluated.save_dir), str(config_path), str(data_yaml)]
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
        },
        metrics=results,
        artifacts=artifacts,
        model_path=model_path,
        extra_tags={"evaluation_split": split},
        actual_device=selected_device,
    )
    report = {
        "schema_version": 1,
        "model_path": str(model_file),
        "model_sha256": file_sha256(model_file),
        "data_config_sha256": file_sha256(data_yaml),
        "config_sha256": file_sha256(config_path),
        "evaluation_provenance": provenance(),
        "split": split,
        "device": selected_device,
        "mlflow_run_id": run_id,
        "metrics": results,
    }
    output = Path(output_path)
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
