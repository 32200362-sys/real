from smart_coaster.behavior.rule_classifier import RuleBehaviorClassifier
from smart_coaster.config import AppConfig, BehaviorConfig, RiskConfig, StateConfig
from smart_coaster.decision.state_machine import StateMachine
from smart_coaster.types import (BehaviorLabel, BehaviorResult, FrameResult, InteractionFeatures,
    Point2D, RiskResult, RobotState)


def interaction(**changes):
    values = dict(hand_key="h", side="Left", hand_position=Point2D(0,0), cup_position=Point2D(100,0),
        hand_velocity=Point2D(50,0), distance=100, distance_rate=-50, closing_speed=50,
        direction_angle=0, ttc_seconds=2.0, data_valid=True)
    values.update(changes)
    return InteractionFeatures(**values)


def test_reaching_holding_and_retracting_robot_policies():
    classifier = RuleBehaviorClassifier(BehaviorConfig(collision_ttc_s=1.0, confirm_frames=1))
    state = StateMachine(StateConfig(), RiskConfig())
    reaching = classifier.classify(interaction(ttc_seconds=2.0), 1)
    assert reaching.label == BehaviorLabel.REACHING
    assert state.update_behavior(reaching, RiskResult(20,"WATCH"), None, 1).state == RobotState.HOLD
    holding = classifier.classify(interaction(distance=40, closing_speed=0, distance_rate=0,
        hand_velocity=Point2D(0,0), ttc_seconds=None), 2)
    assert holding.label == BehaviorLabel.HOLDING
    state.state = RobotState.AVOIDING
    retracting = classifier.classify(interaction(distance_rate=60, closing_speed=0, ttc_seconds=None), 3)
    assert retracting.label == BehaviorLabel.RETRACTING
    assert state.update_behavior(retracting, RiskResult(10,"SAFE"), None, 3).state == RobotState.RETURNING


def test_profile_selection_changes_effective_intervals():
    config = AppConfig()
    pc = config.runtime.select("pc")
    pi = config.runtime.select("pi")
    assert pc.yolo_every_n_frames == 1
    assert pi.yolo_every_n_frames == 3 and config.runtime.profile == "pi"


def test_data_collector_writes_sample_and_event_without_camera():
    import shutil
    from pathlib import Path
    from smart_coaster.config import DataCollectionConfig
    from smart_coaster.data_collection.collector import DataCollector
    output = Path("data/test_collection")
    if output.exists(): shutil.rmtree(output)
    cfg = DataCollectionConfig(True, str(output), .1, .1, .1, 0)
    collector = DataCollector(cfg, output, fps=10, encoder=lambda frame: b"jpeg")
    result = FrameResult(1000, [], None, BehaviorResult(BehaviorLabel.REACHING,.8,"test"))
    collector.collect(object(), result, {"frame":0}, 0)
    collector.collect(object(), result, {"frame":1}, .1)
    collector.close()
    assert (output/"samples/images/1000.jpg").read_bytes() == b"jpeg"
    assert list((output/"events/metadata").glob("*.jsonl"))
    shutil.rmtree(output)
