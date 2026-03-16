from __future__ import annotations

import numpy as np


TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [2, 6, 10, 14, 18]


def detect_demo_sign(landmarks: np.ndarray) -> tuple[str, float]:
    if landmarks is None or landmarks.size == 0:
        return "NO_HAND", 0.0

    points = landmarks.reshape(21, 3)
    extended = []

    thumb_open = points[TIP_IDS[0], 0] > points[PIP_IDS[0], 0]
    extended.append(thumb_open)
    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        extended.append(points[tip_id, 1] < points[pip_id, 1])

    open_count = sum(extended)

    if open_count == 0:
        return "YES", 0.72
    if open_count == 5:
        return "HELLO", 0.75
    if extended[1] and extended[2] and not extended[3] and not extended[4]:
        return "THANK_YOU", 0.71
    if extended[1] and not any(extended[2:]):
        return "NO", 0.71
    if extended[0] and extended[1] and not extended[2] and not extended[3] and extended[4]:
        return "PLEASE", 0.70
    return "UNKNOWN", 0.35
