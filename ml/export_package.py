import json
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO

def export_model(
    model_path: str,
    output_dir: str = "models",
    imgsz: int = 640,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz))

    target_pt = out_dir / "pcb_defect_best.pt"
    target_onnx = out_dir / "pcb_defect_best.onnx"
    alias_onnx = out_dir / "model.onnx"

    if Path(model_path).resolve() != target_pt.resolve():
        shutil.copy2(model_path, target_pt)
    if onnx_path.resolve() != target_onnx.resolve():
        shutil.copy2(onnx_path, target_onnx)
    if onnx_path.resolve() != alias_onnx.resolve():
        shutil.copy2(onnx_path, alias_onnx)

    metadata = {
        "model_name": "DeepPCB-YOLOv8",
        "version": "1.0.0",
        "imgsz": imgsz,
        "classes": list(model.names.values()),
    }

    metadata_file = out_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Export complete: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    export_model(model_path=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
