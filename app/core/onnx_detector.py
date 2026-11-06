import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from app.core.base_detector import BaseDetector


class ONNXDetector(BaseDetector):
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        model_path: str | None = None,
        metadata_path: str | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        infer_cfg = cfg.get("inference", {})

        self.model_path = Path(model_path or infer_cfg.get("model_path", "models/pcb_defect_best.onnx"))
        self.metadata_path = Path(metadata_path or infer_cfg.get("metadata_path", "models/metadata.json"))
        self.conf_threshold = conf_threshold or float(infer_cfg.get("conf_threshold", 0.25))
        self.iou_threshold = iou_threshold or float(infer_cfg.get("iou_threshold", 0.45))

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model dosyasi bulunamadi: {self.model_path}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.imgsz = self.input_shape[2] if len(self.input_shape) >= 4 else 640

        self.classes = self._load_metadata()

    def _load_metadata(self) -> list[str]:
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("classes", [])
        return ["open", "short", "mousebite", "spur", "copper", "pin-hole"]

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        orig_h, orig_w = image.shape[:2]
        scale = min(self.imgsz / orig_w, self.imgsz / orig_h)
        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)

        pad_x = (self.imgsz - new_w) / 2.0
        pad_y = (self.imgsz - new_h) / 2.0

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[int(pad_y):int(pad_y + new_h), int(pad_x):int(pad_x + new_w)] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, pad_x, pad_y

    def forward(self, blob: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        return outputs[0]

    def postprocess(
        self,
        outputs: np.ndarray,
        orig_shape: tuple[int, int],
        scale_info: tuple[float, float, float],
    ) -> list[dict[str, Any]]:
        orig_h, orig_w = orig_shape
        scale, pad_x, pad_y = scale_info

        # YOLOv8 ONNX: (8400, 10)
        predictions = outputs[0].T

        boxes = []
        confidences = []
        class_ids = []

        for pred in predictions:
            scores = pred[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence >= self.conf_threshold:
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

                x1 = (cx - w / 2.0 - pad_x) / scale
                y1 = (cy - h / 2.0 - pad_y) / scale
                box_w = w / scale
                box_h = h / scale

                boxes.append([int(x1), int(y1), int(box_w), int(box_h)])
                confidences.append(confidence)
                class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )

        detections = []
        for idx in indices:
            i = int(idx)
            bx, by, bw, bh = boxes[i]

            x1 = max(0, min(bx, orig_w))
            y1 = max(0, min(by, orig_h))
            x2 = max(0, min(bx + bw, orig_w))
            y2 = max(0, min(by + bh, orig_h))

            cls_id = class_ids[i]
            cls_name = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"

            detections.append({
                "box": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                },
                "confidence": round(float(confidences[i]), 4),
                "class_id": cls_id,
                "class_name": cls_name,
            })

        return detections

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        t0 = time.perf_counter()
        blob, scale, pad_x, pad_y = self.preprocess(image)
        t1 = time.perf_counter()

        outputs = self.forward(blob)
        t2 = time.perf_counter()

        detections = self.postprocess(
            outputs,
            orig_shape=image.shape[:2],
            scale_info=(scale, pad_x, pad_y),
        )
        t3 = time.perf_counter()

        return {
            "detections": detections,
            "total_defects": len(detections),
            "latency_ms": {
                "preprocess": round((t1 - t0) * 1000, 2),
                "inference": round((t2 - t1) * 1000, 2),
                "postprocess": round((t3 - t2) * 1000, 2),
                "total": round((t3 - t0) * 1000, 2),
            },
        }
