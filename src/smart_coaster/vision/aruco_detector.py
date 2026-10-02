from __future__ import annotations

import cv2
import numpy as np

from ..config import VisionConfig
from ..types import MarkerObservation, Point2D


class ArucoDetector:
    def __init__(self, config: VisionConfig) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco가 없습니다. opencv-contrib-python을 설치하세요.")
        if not hasattr(cv2.aruco, config.aruco_dictionary):
            raise ValueError(f"지원하지 않는 ArUco 사전: {config.aruco_dictionary}")

        dictionary_id = getattr(cv2.aruco, config.aruco_dictionary)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self.names = {marker_id: name.upper() for name, marker_id in config.marker_ids.items()}

    def detect(self, frame_bgr: np.ndarray) -> list[MarkerObservation]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters
            )

        if ids is None:
            return []

        observations: list[MarkerObservation] = []
        for marker_corners, marker_id_array in zip(corners, ids):
            marker_id = int(np.asarray(marker_id_array).reshape(-1)[0])
            pts = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            center_xy = pts.mean(axis=0)
            area = float(abs(cv2.contourArea(pts)))
            observations.append(
                MarkerObservation(
                    marker_id=marker_id,
                    name=self.names.get(marker_id, f"ID_{marker_id}"),
                    center=Point2D(float(center_xy[0]), float(center_xy[1])),
                    corners=[Point2D(float(x), float(y)) for x, y in pts],
                    area_px2=area,
                )
            )
        return observations
