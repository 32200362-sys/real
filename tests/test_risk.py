from smart_coaster.config import RiskConfig
from smart_coaster.decision.risk_calculator import RiskCalculator
from smart_coaster.types import MotionFeatures, Point2D


def feature(distance, speed, closing, angle, approaching):
    return MotionFeatures(
        hand_key="Right-0",
        hand_position=Point2D(100, 100),
        target_position=Point2D(200, 200),
        velocity_px_s=Point2D(speed, 0),
        speed_px_s=speed,
        distance_to_target_px=distance,
        distance_rate_px_s=closing,
        direction_angle_deg=angle,
        palm_scale_rate_s=0.2,
        approaching=approaching,
    )


def test_fast_close_approach_has_higher_risk():
    calc = RiskCalculator(RiskConfig())
    safe = calc.calculate(feature(300, 40, 5, 120, False))
    danger = calc.calculate(feature(125, 650, -480, 5, True))
    assert safe.score < 45
    assert danger.score >= 70
    assert danger.level == "AVOID"
