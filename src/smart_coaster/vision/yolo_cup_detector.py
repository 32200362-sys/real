from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import BoundingBox, CupDetection, DetectionSource


class YoloCupDetector:
    def __init__(self, model_path: str | Path, *, confidence: float = 0.4, imgsz: int = 640,
                 device: str = "cpu", max_det: int = 20, backend: Any = None) -> None:
        if backend is None:
            try:
                from ultralytics import YOLO
                backend = YOLO(str(model_path))
            except Exception as exc:
                raise RuntimeError(f"YOLO 모델을 로드할 수 없습니다: {model_path}: {exc}") from exc
        self.backend, self.confidence, self.imgsz, self.device, self.max_det = backend, confidence, imgsz, device, max_det
        names = backend.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        self.cup_class_id = next((int(i) for i, name in items if str(name).lower() == "cup"), None)
        if self.cup_class_id is None:
            raise ValueError("YOLO model names에 'cup' 클래스가 없습니다")

    def detect(self, frame) -> list[CupDetection]:
        results = self.backend.predict(source=frame, conf=self.confidence, imgsz=self.imgsz,
                                       device=self.device, max_det=self.max_det, verbose=False)
        detections = []
        for result in results:
            for class_id, confidence, coords in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
                if int(class_id) != self.cup_class_id:
                    continue
                box = BoundingBox(*(float(v) for v in coords))
                detections.append(CupDetection(box, box.center, float(confidence), int(class_id), "cup", DetectionSource.YOLO))
        return detections
