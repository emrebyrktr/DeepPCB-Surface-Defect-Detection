import os
import shutil
import json
import argparse
from ultralytics import YOLO

def export_model(
    model_path: str,
    output_dir: str = "models",
    imgsz: int = 640,
):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz)

    target_pt = os.path.join(output_dir, "pcb_defect_best.pt")
    target_onnx = os.path.join(output_dir, "model.onnx")

    shutil.copy(model_path, target_pt)
    shutil.copy(onnx_path, target_onnx)

    metadata = {
        "model_name": "DeepPCB-YOLOv8",
        "imgsz": imgsz,
        "classes": list(model.names.values()),
    }

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Export complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    export_model(model_path=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    main()