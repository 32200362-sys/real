from __future__ import annotations

from ..config import RiskConfig
from ..types import MotionFeatures, RiskResult


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


class RiskCalculator:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def calculate(self, feature: MotionFeatures | None) -> RiskResult:
        if feature is None:
            return RiskResult(score=0.0, level="SAFE", components={})

        c = self.config
        distance = _clamp(
            (c.distance_far_px - feature.distance_to_target_px)
            / (c.distance_far_px - c.distance_danger_px)
            * 100.0
        )
        speed = _clamp(feature.speed_px_s / c.speed_max_px_s * 100.0)
        closing = _clamp(-feature.distance_rate_px_s / c.closing_rate_max_px_s * 100.0)
        scale = _clamp(feature.palm_scale_rate_s / c.scale_rate_max_s * 100.0)

        if feature.direction_angle_deg <= c.angle_full_deg:
            direction = 100.0
        elif feature.direction_angle_deg >= c.angle_zero_deg:
            direction = 0.0
        else:
            direction = (
                (c.angle_zero_deg - feature.direction_angle_deg)
                / (c.angle_zero_deg - c.angle_full_deg)
                * 100.0
            )

        components = {
            "distance": distance,
            "speed": speed,
            "direction": direction,
            "closing": closing,
            "scale": scale,
        }
        score = sum(components[name] * c.weights[name] for name in components)

        # 손이 컵 방향으로 접근하지 않으며 아직 위험거리 밖이면 회피 점수까지 올라가지 않게 제한합니다.
        if not feature.approaching and feature.distance_to_target_px > c.distance_danger_px:
            score = min(score, c.warning_threshold - 1.0)

        score = _clamp(score)
        if score >= c.avoid_threshold:
            level = "AVOID"
        elif score >= c.warning_threshold:
            level = "WARNING"
        elif score >= 20.0:
            level = "WATCH"
        else:
            level = "SAFE"
        return RiskResult(score=score, level=level, components=components)
