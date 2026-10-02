from types import SimpleNamespace

import numpy as np

from smart_coaster.config import VisionConfig
from smart_coaster.vision.hand_detector import HandDetector
from smart_coaster.vision.pose_detector import PoseDetector


class AdapterImage:
    def __init__(self, image_format, data):
        self.image_format = image_format
        self.data = data


class FakeLandmarker:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def detect_for_video(self, image, timestamp_ms):
        assert isinstance(image, AdapterImage)
        assert not isinstance(image, np.ndarray)
        assert image.data.flags.c_contiguous
        self.calls.append((image, timestamp_ms))
        return self.result

    def detect(self, *args):
        raise AssertionError("VIDEO detector must not call detect()")


def image_factory(*, image_format, data):
    return AdapterImage(image_format, data)


def test_hand_detector_passes_adapter_image_to_detect_for_video():
    backend = FakeLandmarker(SimpleNamespace(hand_landmarks=[], handedness=[]))
    detector = HandDetector(None, VisionConfig(), backend=backend, image_factory=image_factory, srgb_format="SRGB")
    frame = np.zeros((2, 3, 3), dtype=np.uint8)[:, ::-1]
    frame[0, 0] = [1, 2, 3]
    assert detector.detect(frame, 10) == []
    assert backend.calls[0][1] == 10
    assert backend.calls[0][0].data[0, 0].tolist() == [3, 2, 1]


def test_pose_detector_passes_adapter_image_to_detect_for_video():
    backend = FakeLandmarker(SimpleNamespace(pose_landmarks=[]))
    detector = PoseDetector(None, backend=backend, image_factory=image_factory, srgb_format="SRGB")
    frame = np.zeros((2, 3, 3), dtype=np.uint8)[:, ::-1]
    frame[0, 0] = [4, 5, 6]
    assert detector.detect(frame, 11) == []
    assert backend.calls[0][1] == 11
    assert backend.calls[0][0].data[0, 0].tolist() == [6, 5, 4]
