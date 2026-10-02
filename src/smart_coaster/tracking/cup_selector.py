from __future__ import annotations

from ..types import CupDetection, Point2D


class CupSelector:
    def __init__(self, min_confidence: float) -> None:
        self.min_confidence = min_confidence

    def select(self, detections: list[CupDetection], coaster_center: Point2D | None) -> CupDetection | None:
        if coaster_center is None:
            return None
        valid = [d for d in detections if d.confidence >= self.min_confidence]
        return min(valid, key=lambda d: (d.center.distance_to(coaster_center), -d.confidence), default=None)
