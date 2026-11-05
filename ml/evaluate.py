import os
import argparse
import yaml
from ultralytics import YOLO
from ml.tracker import MLflowTracker

def evaluate_model(
    model_path: str,
    data_yaml: str = "configs/data.yaml",
    config_path: str = "configs/config.yaml",
    tracking_uri: str = None,
):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split="test", name="test")

    results = {
        "test_precision": float(metrics.box.mp),
        "test_recall": float(metrics.box.mr),
        "test_map50": float(metrics.box.map50),
        "test_map50_95": float(metrics.box.map),
        "speed_preprocess_ms": float(metrics.speed.get("preprocess", 0.0)),
        "speed_inference_ms": float(metrics.speed.get("inference", 0.0)),
        "speed_postprocess_ms": float(metrics.speed.get("postprocess", 0.0)),
        "speed_total_ms": float(
            metrics.speed.get("preprocess", 0.0)
            + metrics.speed.get("inference", 0.0)
            + metrics.speed.get("postprocess", 0.0)
        ),
    }

    for class_id, class_name in metrics.names.items():
        results[f"test_map50_{class_name}"] = float(metrics.box.ap50[class_id])
        results[f"test_map50_95_{class_name}"] = float(metrics.box.maps[class_id])

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    save_dir = str(metrics.save_dir)

    artifacts = [save_dir]
    if os.path.exists("models/metadata.json"):
        artifacts.append("models/metadata.json")
    if os.path.exists("models/model.onnx"):
        artifacts.append("models/model.onnx")

    tracker = MLflowTracker(
        experiment_name="DeepPCB_Evaluation",
        tracking_uri=tracking_uri,
    )
    tracker.log_experiment(
        run_name="test_evaluation",
        params=config.get("train", {}),
        metrics=results,
        artifacts=artifacts,
        model_path=model_path,
    )

    print(results)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument("--tracking-uri", type=str, default=None)
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
