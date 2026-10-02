import math

from smart_coaster.behavior.rule_classifier import RuleBehaviorClassifier
from smart_coaster.config import BehaviorConfig, TrackingConfig
from smart_coaster.features.ttc_calculator import calculate_ttc
from smart_coaster.tracking.cup_selector import CupSelector
from smart_coaster.tracking.cup_tracker import CupTracker
from smart_coaster.pipeline import FrameProcessor
from smart_coaster.types import (
    BehaviorLabel,
    BoundingBox,
    CupDetection,
    DetectionSource,
    InteractionFeatures,
    Point2D,
)


def detection(x1, y1, x2, y2, confidence=0.8):
    box = BoundingBox(x1, y1, x2, y2)
    return CupDetection(box, box.center, confidence, 4, "cup", DetectionSource.YOLO)


def test_selector_requires_coaster_and_breaks_ties_by_confidence():
    selector = CupSelector(0.5)
    low, high = detection(0, 0, 10, 10, 0.7), detection(10, 0, 20, 10, 0.9)
    assert selector.select([low], None) is None
    assert selector.select([low, high], Point2D(10, 5)) is high


def test_tracker_keeps_id_predicts_then_expires():
    tracker = CupTracker(TrackingConfig(max_missed_frames=1, max_center_distance_px=30))
    first = tracker.update(detection(0, 0, 10, 10), timestamp_s=0.0)
    second = tracker.update(detection(5, 0, 15, 10), timestamp_s=1.0)
    predicted = tracker.update(None, timestamp_s=2.0)
    assert first.track_id == second.track_id == predicted.track_id
    assert predicted.predicted and predicted.source == DetectionSource.TRACK
    assert tracker.update(None, timestamp_s=3.0) is None


def test_ttc_is_finite_or_none():
    assert calculate_ttc(100, 50, 10) == 2.0
    assert calculate_ttc(100, 0, 10) is None
    assert calculate_ttc(math.nan, 50, 10) is None


def feature(**changes):
    values = dict(
        hand_key="Left-0", side="Left", hand_position=Point2D(0, 0),
        cup_position=Point2D(100, 0), hand_velocity=Point2D(100, 0),
        distance=100, distance_rate=-100, closing_speed=100,
        direction_angle=0, ttc_seconds=1.0, data_valid=True,
    )
    values.update(changes)
    return InteractionFeatures(**values)


def test_rule_classifier_prioritizes_unknown_and_classifies_collision():
    classifier = RuleBehaviorClassifier(BehaviorConfig(confirm_frames=1))
    assert classifier.classify(feature(data_valid=False), 0).label == BehaviorLabel.UNKNOWN
    assert classifier.classify(feature(), 1).label == BehaviorLabel.COLLISION_RISK


def test_frame_processor_is_testable_with_fake_backends():
    class Detector:
        def detect(self, frame): return [detection(90, 90, 110, 110)]
    processor = FrameProcessor(Detector(), CupSelector(0.4), CupTracker(TrackingConfig()),
                               RuleBehaviorClassifier(BehaviorConfig(confirm_frames=1)))
    result = processor.process_frame(object(), 1000, 1.0, coaster_center=Point2D(100, 100),
                                     interaction=feature())
    assert result.cup_track is not None
    assert result.behavior.label == BehaviorLabel.COLLISION_RISK
