from __future__ import annotations

from pathlib import Path


class PoseDetector:
    """MediaPipe Pose VIDEO adapter with an injectable backend for hardware-free tests."""

    def __init__(self, model_path: str | Path | None, *, backend=None, image_factory=None,
                 srgb_format=None) -> None:
        self.image_factory, self.srgb_format = image_factory, srgb_format
        if backend is not None:
            self.landmarker = backend
            return
        assert model_path is not None
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"MediaPipe pose 모델이 없습니다: {path}")
        import mediapipe as mp
        # Read the model into memory instead of passing a path: MediaPipe's C++
        # loader cannot open non-ASCII (e.g. Korean) Windows paths.
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=path.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        close = getattr(self.landmarker, "close", None)
        if close:
            close()

    def detect(self, frame_bgr, timestamp_ms: int):
        from ..types import ArmObservation, PersonObservation, Point2D
        from .mediapipe_image import bgr_to_mp_image
        height, width = frame_bgr.shape[:2]
        mp_image = bgr_to_mp_image(frame_bgr, image_factory=self.image_factory, srgb_format=self.srgb_format)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        people = []
        for person_id, landmarks in enumerate(result.pose_landmarks):
            def point(index):
                lm = landmarks[index]
                return Point2D(lm.x * width, lm.y * height)
            arms = []
            for side, indices in (("Left", (11, 13, 15)), ("Right", (12, 14, 16))):
                confidence = min(float(landmarks[i].visibility or 0) for i in indices)
                arms.append(ArmObservation(person_id, side, point(indices[0]), point(indices[1]), point(indices[2]), confidence))
            center = Point2D(sum(a.shoulder.x for a in arms) / 2, sum(a.shoulder.y for a in arms) / 2)
            people.append(PersonObservation(person_id, arms, center, min(a.confidence for a in arms)))
        return people
