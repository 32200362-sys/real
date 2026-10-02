from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "markers"
MARKERS = {10: "CUP", 20: "COASTER", 30: "LAPTOP", 40: "PAPER"}
SIZE = 800
BORDER = 120


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    for marker_id, name in MARKERS.items():
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, SIZE)
        else:
            marker = np.zeros((SIZE, SIZE), dtype=np.uint8)
            cv2.aruco.drawMarker(dictionary, marker_id, SIZE, marker, 1)

        canvas = np.full((SIZE + 2 * BORDER, SIZE + 2 * BORDER), 255, dtype=np.uint8)
        canvas[BORDER:BORDER + SIZE, BORDER:BORDER + SIZE] = marker
        path = OUT / f"marker_{marker_id}_{name}.png"
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError(f"마커 저장 실패: {path}")
        print(path)


if __name__ == "__main__":
    main()
