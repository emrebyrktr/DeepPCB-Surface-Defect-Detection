import argparse
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_model(
    config_path: str = "configs/config.yaml", *, model_factory: Callable[..., Any] | None = None
) -> Any:
    config = load_config(config_path)

    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    paths_cfg = config.get("paths", {})

    if model_factory is None:
        from ultralytics import YOLO

        model_factory = YOLO
    model = model_factory(model_cfg.get("type", "yolov8n.pt"))

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

    checkpoint = getattr(getattr(model, "trainer", None), "best", None)
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Training did not produce a best checkpoint: {checkpoint!r}")

    best_weight_path = Path(checkpoint)
    target_path = Path(output_model_dir) / "pcb_defect_best.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if best_weight_path.resolve() != target_path.resolve():
        temporary_path = None
        try:
            with NamedTemporaryFile(
                dir=target_path.parent, prefix=f".{target_path.name}-", suffix=".tmp", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            shutil.copy2(best_weight_path, temporary_path)
            temporary_path.replace(target_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    logger.info("Model saved to %s", target_path)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    train_model(config_path=args.config)


if __name__ == "__main__":
    main()
