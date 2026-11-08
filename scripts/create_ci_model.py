"""Create a valid, deterministic ONNX fixture for CI; never a trained PCB model."""

import argparse
import hashlib
import json
from pathlib import Path


def create_ci_model(output_dir: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    model_path = output_dir / "pcb_defect_best.onnx"
    metadata_path = output_dir / "metadata.json"
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing model package.")

    # YOLOv8-compatible output: one centered box and six class scores.
    prediction = helper.make_tensor(
        "prediction", TensorProto.FLOAT, [1, 10, 1],
        [320.0, 320.0, 64.0, 64.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output0"], value=prediction)],
        "pcb-ci-fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 10, 1])],
    )
    model = helper.make_model(
        graph, producer_name="pcb-ci-fixture", opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, model_path)
    metadata = {
        "model_name": "ci-fixture",
        "version": "1.0.0",
        "imgsz": 640,
        "classes": ["open", "short", "mousebite", "spur", "copper", "pin-hole"],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "purpose": "CI runtime smoke test; not suitable for PCB evaluation or deployment",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Created CI-only model package in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    create_ci_model(args.output_dir)


if __name__ == "__main__":
    main()
