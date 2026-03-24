from typing import Tuple

import numpy as np


EARLY_LAYERS = slice(0, 9)
MIDDLE_LAYERS = slice(9, 18)
LATE_LAYERS = slice(18, 27)


def extract_discrepancy_features(vectors: np.ndarray) -> Tuple[np.ndarray, list]:
    mean_early = vectors[:, EARLY_LAYERS].mean(axis=1)
    mean_mid = vectors[:, MIDDLE_LAYERS].mean(axis=1)
    mean_late = vectors[:, LATE_LAYERS].mean(axis=1)

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
    )
    names = [
        "mean_early",
        "mean_mid",
        "mean_late",
        "diff_mid_early",
        "diff_late_mid",
        "diff_late_early",
        "slope",
    ]
    return features.astype(np.float32), names


def detect_change_points(vector: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    d1 = np.diff(vector)
    abs_d1 = np.abs(d1)
    threshold = abs_d1.mean() + sigma * abs_d1.std()
    return abs_d1 > threshold


def extract_change_point_features(vectors: np.ndarray) -> Tuple[np.ndarray, list]:
    n_samples = vectors.shape[0]
    features = np.zeros((n_samples, 8), dtype=np.float32)
    for index, vector in enumerate(vectors):
        d1 = np.diff(vector)
        d2 = np.diff(d1)
        abs_d1 = np.abs(d1)
        abs_d2 = np.abs(d2)

        cp1 = detect_change_points(vector, sigma=2.0)
        cp2_threshold = abs_d2.mean() + 2.0 * abs_d2.std() if abs_d2.size else 0.0
        cp2 = abs_d2 > cp2_threshold if abs_d2.size else np.array([], dtype=bool)

        features[index, 0] = float(cp1.sum())
        features[index, 1] = float(cp2.sum())
        features[index, 2] = float(np.argmax(cp1)) if cp1.any() else -1.0
        features[index, 3] = float(np.argmax(abs_d1)) if abs_d1.size else -1.0
        features[index, 4] = float(abs_d1.max()) if abs_d1.size else 0.0
        features[index, 5] = float(abs_d1.mean()) if abs_d1.size else 0.0
        features[index, 6] = float(d1.var()) if d1.size else 0.0
        features[index, 7] = float(abs_d2.max()) if abs_d2.size else 0.0

    names = [
        "n_cp_d1",
        "n_cp_d2",
        "first_cp_loc",
        "max_change_loc",
        "max_change_mag",
        "mean_change_mag",
        "roughness",
        "max_sharpness",
    ]
    return features, names
