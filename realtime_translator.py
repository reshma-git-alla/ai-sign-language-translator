from __future__ import annotations

import sys
from collections import deque

import cv2
import numpy as np

from src.sign_translator.config import MODELS_DIR, SEQUENCE_LENGTH, ensure_project_dirs, load_labels
from src.sign_translator.inference import PredictionEngine
from src.sign_translator.landmarks import HandTracker


def draw_panel(frame: np.ndarray, predicted_label: str, confidence: float, sentence: str, trained: bool) -> np.ndarray:
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 100), (12, 18, 28), -1)
    mode = "Trained model" if trained else "Heuristic demo mode"
    cv2.putText(panel, mode, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 180), 2)
    color = (255, 255, 255) if predicted_label not in {"UNCERTAIN", "UNKNOWN"} else (0, 190, 255)
    cv2.putText(
        panel,
        f"Prediction: {predicted_label} ({confidence:.2f})",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )
    cv2.putText(panel, sentence, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    return panel


def main() -> int:
    ensure_project_dirs()
    labels = load_labels()
    model_path = MODELS_DIR / "sign_translator.keras"

    tracker = HandTracker(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    engine = PredictionEngine(model_path if model_path.exists() else None, labels)
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Could not open the webcam.", file=sys.stderr)
        return 1

    sequence: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Webcam frame could not be read.", file=sys.stderr)
                return 1

            detection = tracker.process(frame)
            predicted_label = "NO_HAND"
            confidence = 0.0

            if detection.landmarks is not None:
                sequence.append(detection.landmarks)
                if len(sequence) == SEQUENCE_LENGTH:
                    predicted_label, confidence = engine.predict(
                        np.asarray(sequence, dtype=np.float32),
                        detection.landmarks,
                    )

            panel = draw_panel(
                detection.frame,
                predicted_label,
                confidence,
                engine.sentence_text(),
                engine.is_trained,
            )
            cv2.imshow("AI Sign Language Translator", panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                engine.sentence.clear()
                engine.history.clear()
                sequence.clear()

        return 0
    finally:
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
