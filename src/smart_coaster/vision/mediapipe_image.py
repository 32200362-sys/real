from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


def bgr_to_mp_image(frame_bgr: np.ndarray, *, image_factory=None, srgb_format=None):
    """Convert one OpenCV BGR frame at the MediaPipe detector boundary."""
    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    factory = image_factory or mp.Image
    return factory(image_format=srgb_format or mp.ImageFormat.SRGB, data=rgb_frame)
