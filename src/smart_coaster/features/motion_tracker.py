from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot

from ..config import MotionConfig
from ..types import HandObservation, MotionFeatures, Point2D


@dataclass
class _TrackState:
    position: Point2D
    velocity: Point2D
    palm_scale_px: float
    distance_px: float
    timestamp_s: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class MotionTracker:
    def __init__(self, config: MotionConfig) -> None:
        self.config = config
        self._states: dict[str, _TrackState] = {}

    def update_all(
        self,
        hands: list[HandObservation],
        target: Point2D | None,
        timestamp_s: float,
    ) -> list[MotionFeatures]:
        if target is None:
            self._remove_stale(timestamp_s)
            return []

        features = [self._update_one(hand, target, timestamp_s) for hand in hands]
        self._remove_stale(timestamp_s, active={hand.key for hand in hands})
        return features

    def _update_one(
        self,
        hand: HandObservation,
        target: Point2D,
        timestamp_s: float,
    ) -> MotionFeatures:
        previous = self._states.get(hand.key)
        distance = hand.center.distance_to(target)

        if previous is None:
            state = _TrackState(
                position=hand.center,
                velocity=Point2D(0.0, 0.0),
                palm_scale_px=hand.palm_scale_px,
                distance_px=distance,
                timestamp_s=timestamp_s,
            )
            self._states[hand.key] = state
            return MotionFeatures(
                hand_key=hand.key,
                hand_position=hand.center,
                target_position=target,
                velocity_px_s=Point2D(0.0, 0.0),
                speed_px_s=0.0,
                distance_to_target_px=distance,
                distance_rate_px_s=0.0,
                direction_angle_deg=180.0,
                palm_scale_rate_s=0.0,
                approaching=False,
            )

        dt = _clamp(
            timestamp_s - previous.timestamp_s,
            self.config.min_dt_s,
            self.config.max_dt_s,
        )
        a = self.config.smoothing_alpha
        position = Point2D(
            a * hand.center.x + (1.0 - a) * previous.position.x,
            a * hand.center.y + (1.0 - a) * previous.position.y,
        )
        raw_velocity = Point2D(
            (position.x - previous.position.x) / dt,
            (position.y - previous.position.y) / dt,
        )
        va = self.config.velocity_smoothing_alpha
        velocity = Point2D(
            va * raw_velocity.x + (1.0 - va) * previous.velocity.x,
            va * raw_velocity.y + (1.0 - va) * previous.velocity.y,
        )
        speed = hypot(velocity.x, velocity.y)
        distance = position.distance_to(target)
        distance_rate = (distance - previous.distance_px) / dt
        scale_rate = (
            (hand.palm_scale_px - previous.palm_scale_px)
            / max(previous.palm_scale_px, 1.0)
            / dt
        )

        to_target_x = target.x - position.x
        to_target_y = target.y - position.y
        to_target_norm = hypot(to_target_x, to_target_y)
        if speed < 1e-6 or to_target_norm < 1e-6:
            angle = 180.0
        else:
            cosine = (velocity.x * to_target_x + velocity.y * to_target_y) / (
                speed * to_target_norm
            )
            angle = degrees(acos(_clamp(cosine, -1.0, 1.0)))

        approaching = distance_rate < -20.0 and angle < 75.0
        self._states[hand.key] = _TrackState(
            position=position,
            velocity=velocity,
            palm_scale_px=hand.palm_scale_px,
            distance_px=distance,
            timestamp_s=timestamp_s,
        )

        return MotionFeatures(
            hand_key=hand.key,
            hand_position=position,
            target_position=target,
            velocity_px_s=velocity,
            speed_px_s=speed,
            distance_to_target_px=distance,
            distance_rate_px_s=distance_rate,
            direction_angle_deg=angle,
            palm_scale_rate_s=scale_rate,
            approaching=approaching,
        )

    def _remove_stale(self, timestamp_s: float, active: set[str] | None = None) -> None:
        active = active or set()
        stale = [
            key
            for key, state in self._states.items()
            if key not in active and timestamp_s - state.timestamp_s > self.config.stale_track_s
        ]
        for key in stale:
            self._states.pop(key, None)
