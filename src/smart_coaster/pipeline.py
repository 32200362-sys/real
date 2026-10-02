from __future__ import annotations

from .behavior.base import BehaviorClassifier
from .tracking.cup_selector import CupSelector
from .tracking.cup_tracker import CupTracker
from .types import BehaviorLabel, BehaviorResult, FrameResult, InteractionFeatures, Point2D
from .features.interaction_analyzer import build_interaction
from .types import BoundingBox, CupDetection, DetectionSource


class FrameProcessor:
    """Pure orchestration boundary; camera, UI and UDP stay in app.py."""

    def __init__(self, cup_detector, cup_selector: CupSelector, cup_tracker: CupTracker,
                 behavior_classifier: BehaviorClassifier, *, pose_detector=None,
                 hand_detector=None, person_selector=None, motion_tracker=None,
                 profile=None, fallback_enabled: bool = True, fallback_marker_id: int = 10,
                 min_closing_speed: float = 40.0, coaster_marker_id: int = 20,
                 hand_only_fallback: bool = False) -> None:
        self.cup_detector = cup_detector
        self.cup_selector = cup_selector
        self.cup_tracker = cup_tracker
        self.behavior_classifier = behavior_classifier
        self.pose_detector, self.hand_detector = pose_detector, hand_detector
        self.person_selector, self.motion_tracker = person_selector, motion_tracker
        self.profile = profile
        self.fallback_enabled, self.fallback_marker_id = fallback_enabled, fallback_marker_id
        self.min_closing_speed = min_closing_speed
        self.coaster_marker_id = coaster_marker_id
        # 팔(Pose)이 매칭되지 않을 때 컵에 가장 가까운 손만으로 판단한다.
        self.hand_only_fallback = hand_only_fallback
        self._last_detections = []
        self._last_people = []
        self._last_hands = []

    def process_frame(self, frame, timestamp_ms: int, now_s: float, *,
                      coaster_center: Point2D | None,
                      interaction: InteractionFeatures | None = None) -> FrameResult:
        detections = self.cup_detector.detect(frame)
        selected = self.cup_selector.select(detections, coaster_center)
        track = self.cup_tracker.update(selected, now_s) if coaster_center is not None else None
        if coaster_center is None or track is None or interaction is None:
            behavior = BehaviorResult(BehaviorLabel.UNKNOWN, 1.0, "코스터/컵/사람 관측 부족")
        else:
            behavior = self.behavior_classifier.classify(interaction, now_s)
        return FrameResult(timestamp_ms, detections, track, behavior, interaction)

    def process_observations(self, frame, timestamp_ms: int, now_s: float, frame_index: int,
                             markers: list) -> FrameResult:
        by_id = {m.marker_id: m for m in markers}
        coaster = by_id.get(self.coaster_marker_id)
        coaster_center = coaster.center if coaster else None
        yolo_interval = self.profile.yolo_every_n_frames if self.profile else 1
        ran_yolo = frame_index % yolo_interval == 0
        if ran_yolo:
            self._last_detections = self.cup_detector.detect(frame)
            selected = self.cup_selector.select(self._last_detections, coaster_center)
            track = self.cup_tracker.update(selected, now_s) if coaster_center else None
        else:
            track = self.cup_tracker.update(None, now_s) if coaster_center else None
        fallback = by_id.get(self.fallback_marker_id)
        if track is None and coaster_center and fallback and self.fallback_enabled:
            xs, ys = [p.x for p in fallback.corners], [p.y for p in fallback.corners]
            box = BoundingBox(min(xs), min(ys), max(xs), max(ys))
            detection = CupDetection(box, fallback.center, 1.0, -1, "cup", DetectionSource.ARUCO_FALLBACK)
            track = self.cup_tracker.update(detection, now_s)

        pose_interval = self.profile.pose_every_n_frames if self.profile else 1
        hand_interval = self.profile.hand_every_n_frames if self.profile else 1
        if self.pose_detector and frame_index % pose_interval == 0:
            self._last_people = self.pose_detector.detect(frame, timestamp_ms)
        if self.hand_detector and frame_index % hand_interval == 0:
            self._last_hands = self.hand_detector.detect(frame, timestamp_ms)
        person = self.person_selector.select(self._last_people, coaster_center) if self.person_selector else None
        matches = self.person_selector.match_hands(person, self._last_hands) if person and self.person_selector else {}
        matched_hand = next(iter(matches.values()), None)
        arm = next((a for a in person.arms if a.side in matches), None) if person else None
        hand_only = False
        if matched_hand is None and self.hand_only_fallback and self._last_hands and track is not None:
            matched_hand = min(self._last_hands, key=lambda h: h.center.distance_to(track.center))
            arm, hand_only = None, True
        motions = self.motion_tracker.update_all([matched_hand] if matched_hand else [], track.center if track else None, now_s) if self.motion_tracker else []
        motion = motions[0] if motions else None
        interaction = build_interaction(motion, matched_hand, arm, self.min_closing_speed,
                                        allow_hand_only=hand_only)
        if coaster_center is None or track is None or interaction is None:
            behavior = BehaviorResult(BehaviorLabel.UNKNOWN, 1.0, "코스터/컵/사람 관측 부족")
        else:
            behavior = self.behavior_classifier.classify(interaction, now_s)
        return FrameResult(timestamp_ms, list(self._last_detections), track, behavior, interaction,
                           list(self._last_hands), list(self._last_people), person, motion)
