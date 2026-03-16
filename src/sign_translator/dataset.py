from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SEQUENCE_LENGTH


def build_sequence_dir(label: str) -> Path:
    directory = RAW_DATA_DIR / label
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def next_sequence_index(label: str) -> int:
    label_dir = build_sequence_dir(label)
    existing = [int(path.stem.split("_")[-1]) for path in label_dir.glob("sequence_*.npy")]
    return max(existing, default=-1) + 1


def save_sequence(label: str, frames: list[np.ndarray]) -> Path:
    if len(frames) != SEQUENCE_LENGTH:
        raise ValueError(f"Expected {SEQUENCE_LENGTH} frames, received {len(frames)}.")

    index = next_sequence_index(label)
    path = build_sequence_dir(label) / f"sequence_{index:03d}.npy"
    np.save(path, np.asarray(frames, dtype=np.float32))
    return path


def load_dataset(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[int] = []

    for index, label in enumerate(labels):
        for sequence_file in sorted((RAW_DATA_DIR / label).glob("sequence_*.npy")):
            sequence = np.load(sequence_file)
            if sequence.shape != (SEQUENCE_LENGTH, sequence.shape[1]):
                # Accept saved landmark dimension as long as the sequence length is stable.
                if sequence.shape[0] != SEQUENCE_LENGTH:
                    continue
            sequences.append(sequence.astype(np.float32))
            targets.append(index)

    if not sequences:
        raise FileNotFoundError(
            f"No training sequences found in {RAW_DATA_DIR}. Collect data before training."
        )

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def create_splits(
    features: np.ndarray, labels: np.ndarray, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )


def save_training_metadata(history: dict, labels: list[str]) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = PROCESSED_DATA_DIR / "training_history.json"
    metadata_path.write_text(
        json.dumps({"labels": labels, "history": history}, indent=2),
        encoding="utf-8",
    )
    return metadata_path

