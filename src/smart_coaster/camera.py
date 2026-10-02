from __future__ import annotations

from pathlib import Path

import cv2

from .config import CameraConfig


class CameraSource:
    def __init__(self, source: int | str, config: CameraConfig) -> None:
        self.source = source
        self.config = config
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        source = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if isinstance(source, str) and not Path(source).exists():
            raise FileNotFoundError(f"동영상 파일을 찾을 수 없습니다: {source}")

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라/영상 소스를 열 수 없습니다: {source}")

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        if self.cap is None:
            raise RuntimeError("CameraSource.open() must be called first")
        ok, frame = self.cap.read()
        if ok and self.config.mirror:
            frame = cv2.flip(frame, 1)
        return ok, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
