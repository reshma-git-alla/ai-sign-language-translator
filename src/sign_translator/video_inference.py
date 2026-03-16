from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from tempfile import NamedTemporaryFile

import imageio.v3 as iio
import numpy as np

from .config import MODELS_DIR, SEQUENCE_LENGTH, ensure_project_dirs, load_labels
from .inference import PredictionEngine
from .landmarks import HandTracker


def _write_temp_video(upload_bytes: bytes, suffix: str) -> Path:
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(upload_bytes)
        return Path(temp_file.name)


def analyze_video_bytes(upload_bytes: bytes, suffix: str = ".mp4") -> dict:
    ensure_project_dirs()
    labels = load_labels()
    model_path = MODELS_DIR / "sign_translator.keras"
    video_path = _write_temp_video(upload_bytes, suffix)

    tracker = HandTracker(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    engine = PredictionEngine(model_path if model_path.exists() else None, labels)
    sequence: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    processed_frames = 0
    detected_frames = 0
    predictions: list[dict] = []
    preview_frames: list[np.ndarray] = []

    try:
        for frame in iio.imiter(video_path):
            processed_frames += 1
            frame_rgb = np.asarray(frame, dtype=np.uint8)
            if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
                continue

            detection = tracker.process_rgb(frame_rgb[:, :, :3])
            if detection.landmarks is None:
                continue

            detected_frames += 1
            sequence.append(detection.landmarks)
            if processed_frames % 20 == 0 and len(preview_frames) < 4:
                preview_frames.append(detection.frame)

            if len(sequence) < SEQUENCE_LENGTH:
                continue

            label, confidence = engine.predict(
                np.asarray(sequence, dtype=np.float32),
                detection.landmarks,
            )
            if label not in {"UNCERTAIN", "UNKNOWN", "NO_HAND"}:
                predictions.append(
                    {
                        "frame": processed_frames,
                        "label": label,
                        "confidence": round(confidence, 3),
                    }
                )
    finally:
        tracker.close()
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass

    label_counter = Counter(item["label"] for item in predictions)
    dominant_label, dominant_count = ("NO_PREDICTION", 0)
    if label_counter:
        dominant_label, dominant_count = label_counter.most_common(1)[0]

    confidence_values = [item["confidence"] for item in predictions]
    average_confidence = float(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0

    return {
        "processed_frames": processed_frames,
        "detected_frames": detected_frames,
        "predictions": predictions,
        "sentence": engine.sentence_text(),
        "dominant_label": dominant_label,
        "dominant_count": dominant_count,
        "average_confidence": average_confidence,
        "preview_frames": preview_frames,
        "is_trained": engine.is_trained,
    }
