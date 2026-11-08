import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ml.tracker import file_sha256, provenance


def _load_yolo(model_path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path)


def export_model(
    model_path: str,
    output_dir: str = "models",
    imgsz: int | None = None,
    config_path: str = "configs/config.yaml",
    model_version: str = "1.0.0",
    training_run_id: str | None = None,
) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    size = imgsz if imgsz is not None else config.get("model", {}).get("imgsz", 640)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("imgsz must be a positive integer")
    if not model_version.strip():
        raise ValueError("model_version must not be empty")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _load_yolo(model_path)
    onnx_path = Path(model.export(format="onnx", imgsz=size))
    target_pt = out_dir / "pcb_defect_best.pt"
    target_onnx = out_dir / "pcb_defect_best.onnx"
    alias_onnx = out_dir / "model.onnx"
    for source, target in (
        (Path(model_path), target_pt),
        (onnx_path, target_onnx),
        (onnx_path, alias_onnx),
    ):
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)

    names = model.names
    if isinstance(names, dict):
        classes = [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    else:
        classes = list(names)
    metadata = {
        "schema_version": 1,
        "model_name": "DeepPCB-YOLOv8",
        "version": model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "imgsz": size,
        "classes": classes,
        "model_file": target_onnx.name,
        "model_sha256": file_sha256(target_onnx),
        "source_model_sha256": file_sha256(target_pt),
        # Identify this export checkout, not an unverified historic training run.
        "export_provenance": {**provenance(), "config_sha256": file_sha256(config_path)},
        "training_run_id": training_run_id,
        "preprocessing": {
            "name": "letterbox-rgb-nchw",
            "version": 1,
            "implementation": "app/core/onnx_detector.py",
            "implementation_sha256": file_sha256("app/core/onnx_detector.py"),
            "imgsz": size,
            "dtype": "float32",
            "normalization": "divide_by_255",
            "padding_value": 114,
        },
        "config_snapshot": {
            "model": config.get("model", {}),
            "train": config.get("train", {}),
            "inference": config.get("inference", {}),
        },
    }
    # Never carry old evaluation results into a newly exported model package.
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Export complete: {out_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--model-version", default="1.0.0")
    parser.add_argument("--training-run-id", default=None)
    args = parser.parse_args()
    export_model(
        model_path=args.model,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        config_path=args.config,
        model_version=args.model_version,
        training_run_id=args.training_run_id,
    )


if __name__ == "__main__":
    main()
