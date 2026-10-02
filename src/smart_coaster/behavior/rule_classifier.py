from __future__ import annotations

import math

from ..config import BehaviorConfig
from ..types import BehaviorLabel, BehaviorResult, InteractionFeatures


class RuleBehaviorClassifier:
    def __init__(self, config: BehaviorConfig) -> None:
        self.config = config
        self._collision_count = 0

    def classify(self, f: InteractionFeatures, timestamp_s: float) -> BehaviorResult:
        values = (f.distance, f.distance_rate, f.closing_speed, f.direction_angle)
        if not f.data_valid or not all(math.isfinite(v) for v in values):
            self._collision_count = 0
            return BehaviorResult(BehaviorLabel.UNKNOWN, 1.0, "필수 관측값 부족")
        collision = (f.closing_speed >= self.config.min_closing_speed_px_s and
                     f.ttc_seconds is not None and f.ttc_seconds <= self.config.collision_ttc_s and
                     f.direction_angle <= 45)
        self._collision_count = self._collision_count + 1 if collision else 0
        if self._collision_count >= self.config.confirm_frames:
            return BehaviorResult(BehaviorLabel.COLLISION_RISK, 0.9, "짧은 TTC와 빠른 접근")
        speed = (f.hand_velocity.x**2 + f.hand_velocity.y**2) ** 0.5
        if f.distance <= self.config.hold_distance_px and speed <= self.config.hold_speed_px_s:
            return BehaviorResult(BehaviorLabel.HOLDING, 0.75, "컵 근처 저속 정지 추정")
        if f.distance_rate > self.config.min_closing_speed_px_s:
            return BehaviorResult(BehaviorLabel.RETRACTING, 0.8, "컵에서 멀어지는 중")
        if f.closing_speed >= self.config.min_closing_speed_px_s:
            return BehaviorResult(BehaviorLabel.REACHING, 0.75, "컵 방향 접근")
        return BehaviorResult(BehaviorLabel.SAFE, 0.8, "충돌 접근 징후 없음")
