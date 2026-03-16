from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from .config import (
    CONFIDENCE_THRESHOLD,
    CONSENSUS_MIN_COUNT,
    MIN_CONFIDENCE_MARGIN,
    PREDICTION_WINDOW,
    SENTENCE_LIMIT,
)
from .heuristics import detect_demo_sign


class PredictionEngine:
    def __init__(self, model_path: Optional[Path], labels: list[str]) -> None:
        self.labels = labels
        self.history: deque[str] = deque(maxlen=PREDICTION_WINDOW)
        self.confidence_history: deque[float] = deque(maxlen=PREDICTION_WINDOW)
        self.sentence: deque[str] = deque(maxlen=SENTENCE_LIMIT)
        self.model = None
        if model_path and model_path.exists():
            self.model = tf.keras.models.load_model(model_path)

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(self, sequence: np.ndarray, latest_landmarks: np.ndarray) -> tuple[str, float]:
        if self.model is not None:
            probabilities = self.model.predict(sequence[np.newaxis, ...], verbose=0)[0]
            top_indices = np.argsort(probabilities)[::-1]
            class_index = int(top_indices[0])
            label = self.labels[class_index]
            confidence = float(probabilities[class_index])
            second_best = float(probabilities[top_indices[1]]) if len(top_indices) > 1 else 0.0
            margin = confidence - second_best
            if margin < MIN_CONFIDENCE_MARGIN:
                label = "UNCERTAIN"
        else:
            label, confidence = detect_demo_sign(latest_landmarks)

        self.history.append(label)
        self.confidence_history.append(confidence)
        consensus, count = Counter(self.history).most_common(1)[0]
        average_confidence = (
            sum(self.confidence_history) / len(self.confidence_history)
            if self.confidence_history
            else 0.0
        )

        if (
            consensus not in {"UNCERTAIN", "UNKNOWN", "NO_HAND"}
            and count >= CONSENSUS_MIN_COUNT
            and average_confidence >= CONFIDENCE_THRESHOLD
        ):
            if not self.sentence or self.sentence[-1] != consensus:
                self.sentence.append(consensus)

        return label, confidence

    def sentence_text(self) -> str:
        return " ".join(self.sentence) if self.sentence else "Waiting for recognized signs..."
