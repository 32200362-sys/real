from __future__ import annotations

from ..config import TrackingConfig
from ..types import BoundingBox, CupDetection, CupTrack, DetectionSource, Point2D


class CupTracker:
    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self.current: CupTrack | None = None
        self._last_timestamp: float | None = None
        self._next_id = 1

    def reset(self) -> None:
        self.current = None
        self._last_timestamp = None

    def update(self, detection: CupDetection | None, timestamp_s: float) -> CupTrack | None:
        if detection is not None and self.current is not None:
            matched = (self.current.bbox.iou(detection.bbox) >= self.config.min_iou or
                       self.current.center.distance_to(detection.center) <= self.config.max_center_distance_px)
            if not matched:
                self.current = None
        if detection is not None:
            if self.current is None:
                track_id, velocity, age = self._next_id, Point2D(0, 0), 1
                self._next_id += 1
            else:
                dt = max(timestamp_s - (self._last_timestamp or timestamp_s), 1e-6)
                track_id, age = self.current.track_id, self.current.age_frames + 1
                velocity = Point2D((detection.center.x-self.current.center.x)/dt,
                                   (detection.center.y-self.current.center.y)/dt)
            self.current = CupTrack(track_id, detection.bbox, detection.center, detection.confidence,
                                    detection.source, velocity, age)
        elif self.current is not None:
            missed = self.current.missed_frames + 1
            if missed > self.config.max_missed_frames:
                self.current = None
            else:
                dt = max(timestamp_s - (self._last_timestamp or timestamp_s), 0.0)
                dx, dy = self.current.velocity_px_s.x * dt, self.current.velocity_px_s.y * dt
                box = BoundingBox(self.current.bbox.x1+dx, self.current.bbox.y1+dy,
                                  self.current.bbox.x2+dx, self.current.bbox.y2+dy)
                self.current = CupTrack(self.current.track_id, box, box.center, self.current.confidence,
                                        DetectionSource.TRACK, self.current.velocity_px_s,
                                        self.current.age_frames+1, missed, True)
        self._last_timestamp = timestamp_s
        return self.current
