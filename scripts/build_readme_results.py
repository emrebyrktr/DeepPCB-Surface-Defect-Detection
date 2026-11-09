"""Build README figures from a completed test-set evaluation.

Run from the project root with ``python -m scripts.build_readme_results``.
The script uses the report's artifact directory and the exact evaluated model.
"""

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
CLASS_METRICS = {
    "precision": "precision", "recall": "recall", "f1": "f1",
    "ap50": "map50", "ap50_95": "map50_95", "images": "images", "instances": "instances",
}
NATIVE_FIGURES = {
    "confusion_matrix.png": ("confusion_matrix.png",),
    "confusion_matrix_normalized.png": ("confusion_matrix_normalized.png",),
    "pr_curve.png": ("BoxPR_curve.png", "PR_curve.png"),
    "f1_curve.png": ("BoxF1_curve.png", "F1_curve.png"),
}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def public_path(value: str | Path) -> str:
    path = project_path(value)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def read_labels(path: Path, classes: list[str]) -> list[tuple[int, float, float, float, float]]:
    labels = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five YOLO label fields: {path}:{number}")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite YOLO label: {path}:{number}")
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < len(classes):
            raise ValueError(f"Invalid label class: {path}:{number}")
        cx, cy, width, height = values[1:]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid normalized label coordinates: {path}:{number}")
        labels.append((class_id, cx, cy, width, height))
    return labels


def dataset_snapshot(data_root: Path, classes: list[str]):
    image_dir = data_root / "test/images"
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No test images found: {image_dir}")
    labels_by_name = {}
    instances = Counter()
    image_counts = Counter()
    files = []
    for image in images:
        label_path = data_root / "test/labels" / f"{image.stem}.txt"
        labels = read_labels(label_path, classes)
        labels_by_name[image.name] = labels
        instances.update(label[0] for label in labels)
        image_counts.update({label[0] for label in labels})
        files.extend([image, label_path])

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(data_root).as_posix()):
        relative = path.relative_to(data_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)

    snapshot = {
        "root": public_path(data_root), "split": "test", "images": len(images),
        "objects": sum(instances.values()), "sha256": digest.hexdigest(),
        "hash_scope": "Matched test images and labels, sorted by relative path; path length, UTF-8 path, file size, then file bytes.",
        "hash_recorded_at": "figure generation",
        "class_counts": {
            name: {"images": image_counts[index], "instances": instances[index]}
            for index, name in enumerate(classes)
        },
    }
    return images, labels_by_name, snapshot


def metric_value(metrics: dict, name: str) -> float:
    if name not in metrics:
        raise ValueError(f"Evaluation report is missing metric: {name}")
    value = float(metrics[name])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid evaluation metric: {name}")
    return value


def report_metrics(report: dict, classes: list[str], snapshot: dict):
    metrics = report["metrics"]
    aggregate = {
        key: metric_value(metrics, f"test_{source}")
        for key, source in {
            "precision": "precision", "recall": "recall", "macro_f1": "f1",
            "map50": "map50", "map50_95": "map50_95",
        }.items()
    }
    aggregate["timing_ms_per_image"] = {
        stage: metric_value(metrics, f"speed_{stage}_ms")
        for stage in ["preprocess", "inference", "postprocess", "total"]
    }
    per_class = {}
    for name in classes:
        row = {key: metric_value(metrics, f"test_{source}_{name}") for key, source in CLASS_METRICS.items()}
        for key in ["images", "instances"]:
            if row[key] != int(row[key]) or int(row[key]) != snapshot["class_counts"][name][key]:
                raise ValueError(f"Report and test-label counts differ for {name}: {key}")
            row[key] = int(row[key])
        if any(row[key] > 1 for key in ["precision", "recall", "f1", "ap50", "ap50_95"]):
            raise ValueError(f"Class metric outside [0, 1]: {name}")
        per_class[name] = row
    if any(aggregate[key] > 1 for key in ["precision", "recall", "macro_f1", "map50", "map50_95"]):
        raise ValueError("Aggregate metric outside [0, 1]")
    return aggregate, per_class


def plot_class_ap(per_class: dict, destination: Path) -> None:
    names = list(per_class)
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for offset, metric, label, color in [
        (-0.18, "ap50", "AP@0.50", "#2676A8"),
        (0.18, "ap50_95", "AP@0.50:0.95", "#65A679"),
    ]:
        values = [100 * per_class[name][metric] for name in names]
        axis.barh(positions + offset, values, height=0.34, label=label, color=color)
        for position, value in zip(positions + offset, values, strict=True):
            axis.text(value + 0.7, position, f"{value:.1f}", va="center", fontsize=9)
    axis.set(yticks=positions, yticklabels=names, xlim=(0, 108), xlabel="Average precision (%)", title="Per-class average precision · test split")
    axis.invert_yaxis()
    axis.set_xticks(range(0, 101, 20))
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def ground_truth_boxes(labels, width: int, height: int, classes: list[str]) -> list[dict]:
    boxes = []
    for class_id, cx, cy, box_width, box_height in labels:
        boxes.append({
            "class_id": class_id, "class_name": classes[class_id],
            "box": {
                "x1": max(0.0, (cx - box_width / 2) * width),
                "y1": max(0.0, (cy - box_height / 2) * height),
                "x2": min(float(width), (cx + box_width / 2) * width),
                "y2": min(float(height), (cy + box_height / 2) * height),
            },
        })
    return boxes


def draw_boxes(axis, image: np.ndarray, detections: list[dict]) -> None:
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    height, width = image.shape[:2]
    for detection in detections:
        box = detection["box"]
        color = COLORS[detection["class_id"]]
        axis.add_patch(Rectangle((box["x1"], box["y1"]), box["x2"] - box["x1"], box["y2"] - box["y1"], fill=False, edgecolor=color, linewidth=1.8))
        label = detection["class_name"]
        if "confidence" in detection:
            label += f" {detection['confidence']:.2f}"
        align_right = box["x1"] > width * 0.65
        anchor_x = box["x2"] if align_right else box["x1"]
        text_color = "black" if sum(to_rgb(color)) / 3 > 0.55 else "white"
        axis.text(anchor_x, max(12, box["y1"] - 3), label, fontsize=7, color=text_color,
                  ha="right" if align_right else "left", va="bottom", bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2})
    axis.set(xlim=(0, width), ylim=(height, 0))
    axis.axis("off")


def plot_predictions(detector, images: list[Path], labels_by_name: dict, classes: list[str], destination: Path) -> list[dict]:
    indices = list(dict.fromkeys([0, len(images) // 2]))
    figure, axes = plt.subplots(len(indices), 2, figsize=(12, 6 * len(indices)), squeeze=False)
    samples = []
    for row, index in enumerate(indices):
        path = images[index]
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        height, width = image.shape[:2]
        truth = ground_truth_boxes(labels_by_name[path.name], width, height, classes)
        prediction = detector.predict(image)
        draw_boxes(axes[row, 0], image, truth)
        draw_boxes(axes[row, 1], image, prediction["detections"])
        axes[row, 0].set_title(f"{path.name}\nGround truth · {len(truth)} boxes", fontsize=10)
        axes[row, 1].set_title(f"{path.name}\nONNX predictions · {prediction['total_defects']} detections", fontsize=10)
        samples.append({"file": path.name, "sorted_test_index": index, "ground_truth": truth, "prediction": prediction})
    figure.legend(handles=[Patch(color=COLORS[index], label=name) for index, name in enumerate(classes)], loc="lower center", ncol=6, frameon=False)
    figure.suptitle("Held-out PCB examples", fontsize=14)
    figure.text(0.5, 0.04, "Fixed sorted test samples · confidence 0.30 · class-wise NMS IoU 0.45 · CPU", ha="center", fontsize=9)
    figure.subplots_adjust(top=0.92, bottom=0.08, hspace=0.17, wspace=0.04)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return samples


def build_results(report_path: Path, data_root: Path, output: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("split") != "test":
        raise ValueError("README results require an evaluation of the test split")
    model_path = project_path(report["model_path"])
    model_hash = file_sha256(model_path)
    if model_hash != report.get("model_sha256"):
        raise ValueError("Model SHA256 does not match the evaluation report")
    metadata_path = model_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes = metadata.get("classes", [])
    if len(classes) != 6 or not all(isinstance(name, str) and name for name in classes):
        raise ValueError("Expected six class names in model metadata")
    artifacts = project_path(report["artifacts_dir"])
    source_figures = {}
    for destination, candidates in NATIVE_FIGURES.items():
        source = next((artifacts / name for name in candidates if (artifacts / name).is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Missing evaluation figure in {artifacts}: {', '.join(candidates)}")
        source_figures[destination] = source
    images, labels_by_name, snapshot = dataset_snapshot(data_root, classes)
    aggregate, per_class = report_metrics(report, classes, snapshot)

    from app.core.onnx_detector import ONNXDetector

    config_path = PROJECT_ROOT / "configs/config.yaml"
    detector = ONNXDetector(config_path=str(config_path), model_path=str(model_path), metadata_path=str(metadata_path), conf_threshold=0.30, iou_threshold=0.45)
    if detector.device != "cpu" or detector.imgsz != 640:
        raise ValueError("README prediction examples require the 640px CPU model")
    output.mkdir(parents=True, exist_ok=True)
    plot_class_ap(per_class, output / "class_ap.png")
    for destination, source in source_figures.items():
        shutil.copyfile(source, output / destination)
    samples = plot_predictions(detector, images, labels_by_name, classes, output / "predictions.png")
    validation = {key: report.get("validation", {}).get(key) for key in ["split", "device", "imgsz", "batch", "workers", "conf", "iou", "plots"]}
    provenance = {key: report.get("evaluation_provenance", {}).get(key) for key in ["git_commit", "git_dirty", "dvc_data_hash"]}
    summary = {
        "schema_version": 1,
        "source_report": {"path": public_path(report_path), "sha256": file_sha256(report_path)},
        "model": {"path": public_path(model_path), "name": metadata.get("model_name"), "version": metadata.get("version"), "sha256": model_hash, "metadata_sha256": file_sha256(metadata_path)},
        "dataset": snapshot,
        "evaluation_provenance": provenance,
        "evaluation_source_hashes": {key: report.get(key) for key in ["data_config_sha256", "config_sha256", "resolved_data_config_sha256"]},
        "validation": validation,
        "aggregate": aggregate,
        "per_class": per_class,
        "metric_notes": {"macro_f1": "Arithmetic mean of the per-class F1 values returned by validation; not the harmonic mean of mean precision and mean recall."},
        "artifacts_dir": public_path(artifacts),
        "figure_sources": {name: {"path": public_path(path), "sha256": file_sha256(path)} for name, path in source_figures.items()},
        "prediction_settings": {"device": detector.device, "imgsz": int(detector.imgsz), "confidence": detector.conf_threshold, "nms_iou": detector.iou_threshold, "config_sha256": file_sha256(config_path), "detector_source_sha256": file_sha256(PROJECT_ROOT / "app/core/onnx_detector.py"), "latency_scope": "ONNXDetector.predict only; image decoding, startup, and HTTP are excluded."},
        "sample_selection": "Lexicographically sorted test-image indices 0 and len(images)//2, chosen before inference.",
        "samples": samples,
        "generator_source_sha256": file_sha256(Path(__file__)),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("docs/results"))
    args = parser.parse_args(argv)
    try:
        summary = build_results(project_path(args.report), project_path(args.data_root), project_path(args.output))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"README figures written to {public_path(args.output)} ({summary['dataset']['images']} test images).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
