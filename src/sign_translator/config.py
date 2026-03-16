from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
LABELS_PATH = BASE_DIR / "labels.json"

SEQUENCE_LENGTH = 30
LANDMARK_DIMENSION = 21 * 3
DEFAULT_COLLECTIONS_PER_LABEL = 25
COLLECTION_WARMUP_FRAMES = 20
PREDICTION_WINDOW = 12
CONFIDENCE_THRESHOLD = 0.70
SENTENCE_LIMIT = 6
CONSENSUS_MIN_COUNT = 5
MIN_CONFIDENCE_MARGIN = 0.18


def ensure_project_dirs() -> None:
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, METADATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_labels() -> list[str]:
    ensure_project_dirs()
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"labels.json was not found at {LABELS_PATH}. Add your sign labels first."
        )

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels or not all(isinstance(item, str) for item in labels):
        raise ValueError("labels.json must contain a non-empty JSON array of strings.")
    return labels
