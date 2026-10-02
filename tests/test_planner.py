from smart_coaster.config import PlannerConfig
from smart_coaster.decision.avoidance_planner import AvoidancePlanner
from smart_coaster.types import (
    MotionFeatures,
    Point2D,
    RiskResult,
    RobotState,
    StateUpdate,
)


def test_planner_moves_away_from_hand_on_left():
    planner = AvoidancePlanner(PlannerConfig())
    feature = MotionFeatures(
        hand_key="Left-0",
        hand_position=Point2D(100, 200),
        target_position=Point2D(300, 200),
        velocity_px_s=Point2D(300, 0),
        speed_px_s=300,
        distance_to_target_px=200,
        distance_rate_px_s=-200,
        direction_angle_deg=0,
        palm_scale_rate_s=0,
        approaching=True,
    )
    command = planner.plan(
        StateUpdate(RobotState.AVOIDING, "test"),
        feature,
        Point2D(300, 200),
        RiskResult(90, "AVOID"),
        1000,
    )
    assert command.cmd == "MOVE"
    assert command.vx > 0
    assert abs(command.vy) < 1e-6
