import shutil
import logging
from pathlib import Path
from typing import Any, Dict
import argparse
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_model(config_path: str = "configs/config.yaml") -> Any:
    config = load_config(config_path)

    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    paths_cfg = config.get("paths", {})

    model = YOLO(model_cfg.get("type", "yolov8n.pt"))

    project_dir = paths_cfg.get("project_dir", "runs/train")
    experiment_name = paths_cfg.get("experiment_name", "pcb_defect")
    output_model_dir = paths_cfg.get("output_model_dir", "models")

    model.train(
        data=train_cfg.get("data", "configs/data.yaml"),
        epochs=train_cfg.get("epochs", 50),
        imgsz=model_cfg.get("imgsz", 640),
        batch=train_cfg.get("batch_size", 16),
        device=str(train_cfg.get("device", "0")),
        workers=train_cfg.get("workers", 4),
        optimizer=train_cfg.get("optimizer", "auto"),
        lr0=train_cfg.get("lr0", 0.01),
        project=project_dir,
        name=experiment_name,
        save=True,
        verbose=True,
    )

    best_weight_path = Path(project_dir) / experiment_name / "weights" / "best.pt"
    if best_weight_path.exists():
        Path(output_model_dir).mkdir(parents=True, exist_ok=True)
        target_path = Path(output_model_dir) / "pcb_defect_best.pt"
        shutil.copy(best_weight_path, target_path)
        logger.info(f"Model saved to {target_path}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    train_model(config_path=args.config)


if __name__ == "__main__":
    main()
