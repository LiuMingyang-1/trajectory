"""Confidence trajectory feature engineering for Cut A."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ensure_spanlab_importable


ensure_spanlab_importable()


ENT_EARLY = slice(0, 10)
ENT_MIDDLE = slice(10, 19)
ENT_LATE = slice(19, 28)

ICR_EARLY = slice(0, 9)
ICR_MIDDLE = slice(9, 18)
ICR_LATE = slice(18, 27)


def _validate_matrix(vectors: np.ndarray, name: str, expected_dim: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected {name} vectors with shape [N, {expected_dim}], got {matrix.shape}.")
    if matrix.shape[1] != expected_dim:
        raise ValueError(f"Expected {name} vectors with width {expected_dim}, got {matrix.shape[1]}.")
    return matrix


def extract_entropy_discrepancy_features(entropy_vectors: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Apply discrepancy-style feature extraction to 28-layer entropy vectors."""
    vectors = _validate_matrix(entropy_vectors, name="entropy", expected_dim=28)

    mean_early = vectors[:, ENT_EARLY].mean(axis=1)
    mean_mid = vectors[:, ENT_MIDDLE].mean(axis=1)
    mean_late = vectors[:, ENT_LATE].mean(axis=1)

    diff_mid_early = mean_mid - mean_early
    diff_late_mid = mean_late - mean_mid
    diff_late_early = mean_late - mean_early

    layers = np.arange(vectors.shape[1], dtype=np.float32)
    layer_mean = layers.mean()
    layer_var = ((layers - layer_mean) ** 2).sum()
    slopes = ((vectors * (layers[None, :] - layer_mean)).sum(axis=1)) / layer_var

    features = np.column_stack(
        [
            mean_early,
            mean_mid,
            mean_late,
            diff_mid_early,
            diff_late_mid,
            diff_late_early,
            slopes,
        ]
    ).astype(np.float32)
    names = [
        "ent_mean_early",
        "ent_mean_mid",
        "ent_mean_late",
        "ent_diff_mid_early",
        "ent_diff_late_mid",
        "ent_diff_late_early",
        "ent_slope",
    ]
    return features, names


def build_feature_sets(
    icr_vectors: np.ndarray,
    entropy_vectors: np.ndarray,
    delta_entropy_vectors: np.ndarray,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Build Cut A feature-set variants for ablation experiments."""
    icr_matrix = _validate_matrix(icr_vectors, name="ICR", expected_dim=27)
    entropy_matrix = _validate_matrix(entropy_vectors, name="entropy", expected_dim=28)
    delta_entropy_matrix = _validate_matrix(delta_entropy_vectors, name="delta entropy", expected_dim=27)

    from spanlab.features import extract_discrepancy_features as extract_icr_discrepancy_features

    icr_raw = (icr_matrix, [f"icr_{index}" for index in range(icr_matrix.shape[1])])
    entropy_raw = (entropy_matrix, [f"ent_{index}" for index in range(entropy_matrix.shape[1])])
    delta_raw = (delta_entropy_matrix, [f"dent_{index}" for index in range(delta_entropy_matrix.shape[1])])

    icr_disc_features, icr_disc_names = extract_icr_discrepancy_features(icr_matrix)
    entropy_disc_features, entropy_disc_names = extract_entropy_discrepancy_features(entropy_matrix)

    icr_entropy = (
        np.hstack([icr_matrix, entropy_matrix]).astype(np.float32),
        [f"icr_{index}" for index in range(icr_matrix.shape[1])]
        + [f"ent_{index}" for index in range(entropy_matrix.shape[1])],
    )
    icr_delta_entropy = (
        np.hstack([icr_matrix, delta_entropy_matrix]).astype(np.float32),
        [f"icr_{index}" for index in range(icr_matrix.shape[1])]
        + [f"dent_{index}" for index in range(delta_entropy_matrix.shape[1])],
    )
    discrepancy_combined = (
        np.hstack([icr_disc_features, entropy_disc_features]).astype(np.float32),
        [f"icr_{name}" for name in icr_disc_names] + entropy_disc_names,
    )

    return {
        "icr_only": icr_raw,
        "entropy_only": entropy_raw,
        "delta_entropy_only": delta_raw,
        "icr_entropy": icr_entropy,
        "icr_delta_entropy": icr_delta_entropy,
        "discrepancy_combined": discrepancy_combined,
    }


__all__ = [
    "ENT_EARLY",
    "ENT_MIDDLE",
    "ENT_LATE",
    "ICR_EARLY",
    "ICR_MIDDLE",
    "ICR_LATE",
    "extract_entropy_discrepancy_features",
    "build_feature_sets",
]
