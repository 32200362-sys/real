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


def _behavior(label, reason="t"):
    from smart_coaster.types import BehaviorResult
    return BehaviorResult(label, 0.9, reason)


def _drive(labels, **state_kwargs):
    from smart_coaster.config import RiskConfig, StateConfig
    from smart_coaster.decision.state_machine import StateMachine
    from smart_coaster.types import BehaviorLabel, MotionFeatures, Point2D, RiskResult
    machine = StateMachine(StateConfig(**state_kwargs), RiskConfig())
    feature = MotionFeatures("h", Point2D(0, 0), Point2D(50, 0), Point2D(0, 0), 100.0, 50.0, -200.0, 10.0, 0.0, True)
    risk = RiskResult(70.0, "AVOID", {})
    states = []
    for i, label in enumerate(labels):
        states.append(machine.update_behavior(_behavior(label), risk, feature, i * 0.08).state)
    return states


def test_flickering_reaching_between_collision_frames_still_avoids():
    from smart_coaster.types import BehaviorLabel as B, RobotState
    states = _drive([B.REACHING, B.COLLISION_RISK, B.REACHING, B.COLLISION_RISK, B.REACHING, B.COLLISION_RISK])
    assert RobotState.AVOIDING in states


def test_collision_risk_overrides_hold():
    from smart_coaster.types import BehaviorLabel as B, RobotState
    states = _drive([B.HOLDING, B.HOLDING, B.COLLISION_RISK, B.COLLISION_RISK, B.COLLISION_RISK])
    assert states[1] == RobotState.HOLD and states[-1] == RobotState.AVOIDING


def test_grace_zero_keeps_old_strict_behavior_and_plain_reaching_holds():
    from smart_coaster.types import BehaviorLabel as B, RobotState
    strict = _drive([B.COLLISION_RISK, B.REACHING, B.COLLISION_RISK, B.REACHING, B.COLLISION_RISK], avoid_grace_frames=0)
    assert RobotState.AVOIDING not in strict
    assert _drive([B.REACHING, B.REACHING])[-1] == RobotState.HOLD
