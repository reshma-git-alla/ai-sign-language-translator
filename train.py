from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.sign_translator.config import (
    LANDMARK_DIMENSION,
    METADATA_DIR,
    MODELS_DIR,
    SEQUENCE_LENGTH,
    ensure_project_dirs,
    load_labels,
)
from src.sign_translator.dataset import create_splits, load_dataset, save_training_metadata
from src.sign_translator.model import build_sequence_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sign-language sequence model.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_project_dirs()
    labels = load_labels()
    features, targets = load_dataset(labels)

    if features.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"Expected sequence length {SEQUENCE_LENGTH}, got {features.shape[1]}.")
    if features.shape[2] != LANDMARK_DIMENSION:
        raise ValueError(f"Expected landmark dimension {LANDMARK_DIMENSION}, got {features.shape[2]}.")

    x_train, x_test, y_train, y_test = create_splits(features, targets)

    model = build_sequence_model(SEQUENCE_LENGTH, LANDMARK_DIMENSION, len(labels))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "sign_translator.keras"
    metadata_path = METADATA_DIR / "model_info.json"
    model.save(model_path)

    metadata = {
        "labels": labels,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "sequence_length": SEQUENCE_LENGTH,
        "landmark_dimension": LANDMARK_DIMENSION,
        "samples": int(features.shape[0]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_training_metadata(history.history, labels)

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Test accuracy: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

