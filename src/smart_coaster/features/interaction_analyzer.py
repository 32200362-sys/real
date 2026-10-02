from __future__ import annotations

from .ttc_calculator import calculate_ttc
from ..types import ArmObservation, HandObservation, InteractionFeatures, MotionFeatures


def build_interaction(motion: MotionFeatures | None, hand: HandObservation | None,
                      arm: ArmObservation | None, min_closing_speed: float) -> InteractionFeatures | None:
    if motion is None or hand is None:
        return None
    closing = max(0.0, -motion.distance_rate_px_s)
    return InteractionFeatures(
        hand_key=hand.key, side=arm.side if arm else hand.handedness,
        hand_position=motion.hand_position, cup_position=motion.target_position,
        hand_velocity=motion.velocity_px_s, distance=motion.distance_to_target_px,
        distance_rate=motion.distance_rate_px_s, closing_speed=closing,
        direction_angle=motion.direction_angle_deg,
        ttc_seconds=calculate_ttc(motion.distance_to_target_px, closing, min_closing_speed),
        wrist=arm.wrist if arm else None, elbow=arm.elbow if arm else None,
        shoulder=arm.shoulder if arm else None, pose_confidence=arm.confidence if arm else 0.0,
        hand_confidence=hand.confidence, data_valid=arm is not None and hand.confidence > 0,
    )
