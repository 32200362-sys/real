from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from ..config import VisionConfig
from ..types import HandObservation, Point2D, Point3D
from .mediapipe_image import bgr_to_mp_image


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class HandDetector:
    def __init__(self, model_path: Path | None, config: VisionConfig, *, backend=None,
                 image_factory=None, srgb_format=None) -> None:
        self.image_factory, self.srgb_format = image_factory, srgb_format
        if backend is not None:
            self.landmarker = backend
            return
        assert model_path is not None
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe 모델이 없습니다: {model_path}\n"
                "먼저 `python scripts/download_models.py`를 실행하세요."
            )

        # Read the model into memory instead of passing a path: MediaPipe's C++
        # loader cannot open non-ASCII (e.g. Korean) Windows paths.
        base_options = mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes())
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandObservation]:
        height, width = frame_bgr.shape[:2]
        mp_image = bgr_to_mp_image(frame_bgr, image_factory=self.image_factory, srgb_format=self.srgb_format)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        observations: list[HandObservation] = []
        label_counts: defaultdict[str, int] = defaultdict(int)

        for index, landmarks in enumerate(result.hand_landmarks):
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                handedness = category.category_name or "Unknown"
                confidence = float(category.score)
            else:
                handedness = "Unknown"
                confidence = 0.0

            suffix = label_counts[handedness]
            label_counts[handedness] += 1
            key = f"{handedness}-{suffix}"

            points = [Point3D(lm.x * width, lm.y * height, lm.z) for lm in landmarks]
            palm_indices = (0, 5, 9, 13, 17)
            center = Point2D(
                sum(points[i].x for i in palm_indices) / len(palm_indices),
                sum(points[i].y for i in palm_indices) / len(palm_indices),
            )

            xs = [p.x for p in points]
            ys = [p.y for p in points]
            palm_scale = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
            mean_z = sum(p.z for p in points) / len(points)

            observations.append(
                HandObservation(
                    key=key,
                    handedness=handedness,
                    confidence=confidence,
                    center=center,
                    palm_scale_px=palm_scale,
                    mean_z=mean_z,
                    landmarks=points,
                )
            )

        return observations

    def close(self) -> None:
        self.landmarker.close()
