from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


@dataclass
class DetectionResult:
    frame: np.ndarray
    landmarks: Optional[np.ndarray]
    handedness: Optional[str]


class HandTracker:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> DetectionResult:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        annotated = frame.copy()

        if not results.multi_hand_landmarks:
            return DetectionResult(annotated, None, None)

        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            annotated,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        return DetectionResult(
            frame=annotated,
            landmarks=normalize_landmarks(hand_landmarks),
            handedness=handedness,
        )

    def close(self) -> None:
        self.hands.close()


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    origin = coords[0].copy()
    coords -= origin
    max_value = np.max(np.abs(coords))
    if max_value > 0:
        coords /= max_value
    return coords.flatten()

