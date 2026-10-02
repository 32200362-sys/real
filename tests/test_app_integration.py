from types import SimpleNamespace

from smart_coaster.app import SmartCoasterApp
from smart_coaster.behavior.rule_classifier import RuleBehaviorClassifier
from smart_coaster.config import AppConfig, BehaviorConfig, ProfileConfig, StateConfig, TrackingConfig
from smart_coaster.decision.avoidance_planner import AvoidancePlanner
from smart_coaster.decision.risk_calculator import RiskCalculator
from smart_coaster.decision.state_machine import StateMachine
from smart_coaster.features.motion_tracker import MotionTracker
from smart_coaster.pipeline import FrameProcessor
from smart_coaster.tracking.cup_selector import CupSelector
from smart_coaster.tracking.cup_tracker import CupTracker
from smart_coaster.tracking.person_selector import PersonSelector
from smart_coaster.types import (ArmObservation, BoundingBox, CupDetection, DetectionSource,
    HandObservation, MarkerObservation, PersonObservation, Point2D, Point3D, RobotState)


def marker(marker_id, x, y):
    corners = [Point2D(x-5,y-5), Point2D(x+5,y-5), Point2D(x+5,y+5), Point2D(x-5,y+5)]
    return MarkerObservation(marker_id, str(marker_id), Point2D(x,y), corners, 100)


class Sequence:
    def __init__(self, values): self.values, self.i = values, 0
    def detect(self, *args):
        value = self.values[min(self.i, len(self.values)-1)]; self.i += 1; return value


class FakeUdp:
    def __init__(self, enabled): self.enabled, self.sent = enabled, []
    def send(self, command, force=False): self.sent.append(command); return True
    def close(self): pass


class FakeLogger:
    def log_frame(self, frame_index, result, markers, state, risk, command, udp_sent):
        return {"frame_index": frame_index, "behavior": result.behavior.label.value}
    def close(self): pass


class FakeCollector:
    def __init__(self): self.calls=[]
    def collect(self, *args): self.calls.append(args)
    def close(self): pass


def make_app(hand_xs, *, udp=False, yolo_values=None, markers=None):
    config = AppConfig(); config.state = StateConfig(avoid_confirm_frames=1); config.behavior = BehaviorConfig(confirm_frames=1)
    config.runtime.profiles["pc"] = ProfileConfig(); config.runtime.profile = "pc"
    box = BoundingBox(90,90,110,110)
    cups = [CupDetection(box, box.center, .9, 1, "cup", DetectionSource.YOLO)]
    hands = [[HandObservation("Left-0", "Left", .9, Point2D(x,100), 20, 0, [Point3D(x,100,0)]*21)] for x in hand_xs]
    arm = ArmObservation(0,"Left",Point2D(0,100),Point2D(30,100),Point2D(50,100),.9)
    processor = FrameProcessor(Sequence(yolo_values or [cups]), CupSelector(.4), CupTracker(TrackingConfig(max_missed_frames=1)),
        RuleBehaviorClassifier(config.behavior), pose_detector=Sequence([[PersonObservation(0,[arm],Point2D(30,100),.9)]]),
        hand_detector=Sequence(hands), person_selector=PersonSelector(200), motion_tracker=MotionTracker(config.motion),
        profile=config.runtime.profiles["pc"], fallback_marker_id=10, min_closing_speed=40)
    fake_udp, collector = FakeUdp(udp), FakeCollector()
    components = SimpleNamespace(aruco=Sequence([markers or [marker(20,100,100)]]), processor=processor,
        risk=RiskCalculator(config.risk), state=StateMachine(config.state,config.risk), planner=AvoidancePlanner(config.planner),
        udp=fake_udp, logger=FakeLogger(), collector=collector,
        view=SimpleNamespace(render_integrated=lambda frame,*args: frame))
    return SmartCoasterApp(config, 0, components), fake_udp, collector


def test_fast_approach_reaches_avoiding_move_and_calls_collection():
    app, udp, collector = make_app([0,80])
    app.process_frame(object(),1000,0.0); result,state,_,command,_ = app.process_frame(object(),2000,1.0)
    assert result.behavior.label.value == "COLLISION_RISK"
    assert state.state == RobotState.AVOIDING and command.cmd == "MOVE"
    assert not udp.sent and len(collector.calls) == 2


def test_missing_person_is_unknown_stop_and_udp_enabled_preserves_wire():
    app, udp, _ = make_app([0], udp=True)
    app.components.processor._last_people = []
    app.components.processor.pose_detector = Sequence([[]])
    result,state,_,command,_ = app.process_frame(object(),1000,0.0)
    assert result.behavior.label.value == "UNKNOWN" and state.state == RobotState.IDLE and command.cmd == "STOP"
    assert udp.sent and udp.sent[0].to_wire().startswith(b"SC1|")


def test_tracker_prediction_then_aruco_fallback():
    app, _, _ = make_app([0,0,0], yolo_values=[[],[],[]], markers=[marker(20,100,100), marker(10,90,100)])
    result, *_ = app.process_frame(object(),1000,0.0)
    assert result.cup_track.source == DetectionSource.ARUCO_FALLBACK


def hand_only_app(enabled):
    app, udp, collector = make_app([0, 80])
    app.components.processor.pose_detector = Sequence([[]])  # 팔/사람 인식 없음
    app.components.processor.hand_only_fallback = enabled
    return app, udp


def test_hand_only_fallback_avoids_without_pose():
    app, _ = hand_only_app(True)
    app.process_frame(object(), 1000, 0.0)
    result, state, _, command, _ = app.process_frame(object(), 2000, 1.0)
    assert result.behavior.label.value == "COLLISION_RISK"
    assert state.state == RobotState.AVOIDING and command.cmd == "MOVE"
    assert result.interaction.wrist is None and result.interaction.data_valid


def test_without_fallback_missing_pose_stays_unknown_stop():
    app, _ = hand_only_app(False)
    app.process_frame(object(), 1000, 0.0)
    result, state, _, command, _ = app.process_frame(object(), 2000, 1.0)
    assert result.behavior.label.value == "UNKNOWN" and command.cmd == "STOP"


def test_brief_marker_dropout_is_bridged_but_expires():
    app, _, _ = make_app([0, 0, 0, 0], yolo_values=[[]])
    seq = Sequence([[marker(20, 100, 100)], [], [], []])
    app.components.aruco = seq
    app.process_frame(object(), 1000, 0.0)
    app.process_frame(object(), 1100, 0.2)   # 마커 끊김 프레임도 예외 없이 처리
    held = app._hold_recent_markers([], 0.3)
    assert any(m.marker_id == 20 for m in held)
    expired = app._hold_recent_markers([], 1.0)         # 0.5초 초과 -> 제거
    assert not any(m.marker_id == 20 for m in expired)
