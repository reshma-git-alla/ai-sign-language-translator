from __future__ import annotations

import argparse
import sys
from collections import deque

import cv2

from src.sign_translator.config import (
    COLLECTION_WARMUP_FRAMES,
    DEFAULT_COLLECTIONS_PER_LABEL,
    SEQUENCE_LENGTH,
    ensure_project_dirs,
    load_labels,
)
from src.sign_translator.dataset import save_sequence
from src.sign_translator.landmarks import HandTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect sign-language landmark sequences from webcam.")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_COLLECTIONS_PER_LABEL,
        help="Number of sequences to record per label.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = load_labels()
    ensure_project_dirs()

    tracker = HandTracker(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Could not open the webcam.", file=sys.stderr)
        return 1

    try:
        for label in labels:
            print(f"\nCollecting '{label}' samples.")
            print("Press SPACE to start a recording, or Q to quit.")

            collected = 0
            while collected < args.samples:
                sequence: deque = deque(maxlen=SEQUENCE_LENGTH)
                started = False
                warmup = COLLECTION_WARMUP_FRAMES

                while True:
                    ok, frame = capture.read()
                    if not ok:
                        print("Webcam frame could not be read.", file=sys.stderr)
                        return 1

                    detection = tracker.process(frame)
                    display = detection.frame

                    instruction = f"Label: {label} | Sample {collected + 1}/{args.samples}"
                    if not started:
                        prompt = "Press SPACE when ready"
                    elif warmup > 0:
                        prompt = f"Get into pose... {warmup}"
                        warmup -= 1
                    else:
                        prompt = f"Recording {len(sequence)}/{SEQUENCE_LENGTH}"
                        if detection.landmarks is not None:
                            sequence.append(detection.landmarks)
                        if len(sequence) == SEQUENCE_LENGTH:
                            path = save_sequence(label, list(sequence))
                            print(f"Saved {path}")
                            collected += 1
                            break

                    cv2.rectangle(display, (0, 0), (display.shape[1], 70), (20, 20, 20), -1)
                    cv2.putText(display, instruction, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display, prompt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                    cv2.imshow("Collect Sign Data", display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return 0
                    if key == ord(" "):
                        started = True

        print("\nData collection finished.")
        return 0
    finally:
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

