from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from ..config import DataCollectionConfig
from ..types import BehaviorLabel, FrameResult


class DataCollector:
    EVENTS = {BehaviorLabel.REACHING, BehaviorLabel.COLLISION_RISK, BehaviorLabel.HOLDING, BehaviorLabel.UNKNOWN}

    def __init__(self, config: DataCollectionConfig, directory: Path, fps: float, encoder=None) -> None:
        self.config, self.directory, self.fps = config, directory, max(fps, 1.0)
        self.encoder = encoder or self._encode_jpeg
        self.buffer = deque(maxlen=max(1, round(config.pre_event_s * self.fps)))
        self.last_sample_s = float("-inf")
        self.last_event_s = float("-inf")
        self.last_behavior = None
        self.active_event = None
        if config.enabled:
            for path in ("samples/images", "samples/metadata", "events/frames", "events/metadata", "pseudo_labels"):
                (directory / path).mkdir(parents=True, exist_ok=True)

    def collect(self, frame, result: FrameResult, metadata: dict, now_s: float) -> None:
        if not self.config.enabled:
            return
        encoded = self.encoder(frame)
        item = (result.timestamp_ms, encoded, metadata)
        self.buffer.append(item)
        if now_s - self.last_sample_s >= self.config.sample_interval_s:
            self._save_sample(item, result)
            self.last_sample_s = now_s
        entered = result.behavior.label != self.last_behavior and result.behavior.label in self.EVENTS
        if entered and now_s - self.last_event_s >= self.config.event_cooldown_s:
            event_id = f"{result.timestamp_ms}_{result.behavior.label.value}"
            self.active_event = [event_id, round(self.config.post_event_s * self.fps), list(self.buffer)]
            self.last_event_s = now_s
        elif self.active_event:
            self.active_event[2].append(item)
            self.active_event[1] -= 1
            if self.active_event[1] <= 0:
                self._save_event(*self.active_event)
                self.active_event = None
        self.last_behavior = result.behavior.label

    def _save_sample(self, item, result):
        timestamp, encoded, metadata = item
        (self.directory / "samples/images" / f"{timestamp}.jpg").write_bytes(encoded)
        payload = dict(metadata, pseudo_label=bool(result.cup_detections), review_status="unreviewed")
        (self.directory / "samples/metadata" / f"{timestamp}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if result.cup_detections:
            width, height = metadata.get("frame_width"), metadata.get("frame_height")
            if width and height:
                labels = []
                for detection in result.cup_detections:
                    box = detection.bbox
                    labels.append(f"0 {detection.center.x/width:.6f} {detection.center.y/height:.6f} {box.width/width:.6f} {box.height/height:.6f}")
                (self.directory / "pseudo_labels" / f"{timestamp}.txt").write_text("\n".join(labels), encoding="utf-8")

    def _save_event(self, event_id, remaining, items):
        frame_dir = self.directory / "events/frames" / event_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for timestamp, encoded, metadata in items:
            (frame_dir / f"{timestamp}.jpg").write_bytes(encoded)
            lines.append(json.dumps(metadata, ensure_ascii=False))
        (self.directory / "events/metadata" / f"{event_id}.jsonl").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _encode_jpeg(frame) -> bytes:
        import cv2
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return data.tobytes()

    def close(self) -> None:
        if self.active_event:
            self._save_event(*self.active_event)
            self.active_event = None
