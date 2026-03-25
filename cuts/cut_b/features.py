from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ensure_spanlab_importable


ensure_spanlab_importable()


SHALLOW_LAYERS = slice(0, 10)
MID_LAYERS = slice(10, 19)
DEEP_LAYERS = slice(19, 28)
EPS = 1e-6


def _validate_entropy_vectors(entropy_vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(entropy_vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected entropy vectors with shape [N, n_layers], got {vectors.shape}.")
    if vectors.shape[1] < 28:
        raise ValueError(f"Expected at least 28 entropy layers, got {vectors.shape[1]}.")
    return vectors


def _linear_slope(vectors: np.ndarray) -> np.ndarray:
    layers = np.arange(vectors.shape[1], dtype=np.float32)
    centered = layers - layers.mean()
    denom = float((centered**2).sum())
    if denom <= 0:
        return np.zeros(vectors.shape[0], dtype=np.float32)
    return ((vectors * centered[None, :]).sum(axis=1) / denom).astype(np.float32)


def extract_mismatch_features(entropy_vectors: np.ndarray) -> Tuple[np.ndarray, list]:
    """Extract shallow/deep confidence mismatch features from pooled entropy vectors."""
    vectors = _validate_entropy_vectors(entropy_vectors)

    shallow_mean = vectors[:, SHALLOW_LAYERS].mean(axis=1)
    deep_mean = vectors[:, DEEP_LAYERS].mean(axis=1)
    mid_mean = vectors[:, MID_LAYERS].mean(axis=1)
    shallow_deep_ratio = shallow_mean / (deep_mean + EPS)
    shallow_deep_diff = deep_mean - shallow_mean
    entropy_slope = _linear_slope(vectors)

    max_entropy_layer = np.argmax(vectors, axis=1).astype(np.float32)
    if vectors.shape[1] > 1:
        max_entropy_layer = max_entropy_layer / float(vectors.shape[1] - 1)
    else:
        max_entropy_layer = np.zeros(vectors.shape[0], dtype=np.float32)

    features = np.column_stack(
        [
            shallow_mean,
            deep_mean,
            mid_mean,
            shallow_deep_ratio,
            shallow_deep_diff,
            entropy_slope,
            max_entropy_layer,
        ]
    ).astype(np.float32)
    feature_names = [
        "shallow_mean",
        "deep_mean",
        "mid_mean",
        "shallow_deep_ratio",
        "shallow_deep_diff",
        "entropy_slope",
        "max_entropy_layer",
    ]
    return features, feature_names


__all__ = [
    "SHALLOW_LAYERS",
    "MID_LAYERS",
    "DEEP_LAYERS",
    "extract_mismatch_features",
]
