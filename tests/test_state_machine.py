from smart_coaster.config import RiskConfig, StateConfig
from smart_coaster.decision.state_machine import StateMachine
from smart_coaster.types import MotionFeatures, Point2D, RiskResult, RobotState


def make_feature(distance=100, speed=20):
    return MotionFeatures(
        hand_key="Right-0",
        hand_position=Point2D(100, 100),
        target_position=Point2D(150, 100),
        velocity_px_s=Point2D(0, 0),
        speed_px_s=speed,
        distance_to_target_px=distance,
        distance_rate_px_s=0,
        direction_angle_deg=180,
        palm_scale_rate_s=0,
        approaching=False,
    )


def test_slow_near_hand_enters_hold_after_confirmation():
    state = StateMachine(StateConfig(hold_confirm_seconds=0.5), RiskConfig())
    risk = RiskResult(30, "WATCH")
    state.update(risk, make_feature(), True, 1.0)
    result = state.update(risk, make_feature(), True, 1.6)
    assert result.state == RobotState.HOLD
