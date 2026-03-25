from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ensure_spanlab_importable


ensure_spanlab_importable()

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from cut_b.analysis import DEEP_LAYERS, SHALLOW_LAYERS, compute_sample_layer_stats


LABEL_COLORS = {0: "#2166ac", 1: "#b2182b"}
LABEL_NAMES = {0: "Correct", 1: "Hallucinated"}


def _require_matplotlib():
    if plt is None:
        raise RuntimeError("Missing dependency `matplotlib`. Install it with `pip install matplotlib`.")
    return plt


def _ensure_parent(output_path: Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def _layer_mean_vector(entropy_scores: Any) -> np.ndarray:
    matrix = np.asarray(entropy_scores, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected entropy scores with shape [layers, tokens], got {matrix.shape}.")
    if matrix.shape[0] < 28:
        raise ValueError(f"Expected at least 28 entropy layers, got {matrix.shape[0]}.")
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return matrix.mean(axis=1).astype(np.float32)


def _group_layer_means(merged_records: Sequence[dict[str, Any]]) -> dict[int, np.ndarray]:
    grouped: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for record in merged_records:
        label = int(record["label"])
        if label not in grouped:
            continue
        grouped[label].append(_layer_mean_vector(record["entropy_scores"])[:28])

    result: dict[int, np.ndarray] = {}
    for label, rows in grouped.items():
        result[label] = np.vstack(rows) if rows else np.zeros((0, 28), dtype=np.float32)
    return result


def _scatter_points(merged_records: Sequence[dict[str, Any]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
    for record in merged_records:
        label = int(record["label"])
        if label not in grouped:
            continue
        shallow = compute_sample_layer_stats(record["entropy_scores"], SHALLOW_LAYERS)["mean"]
        deep = compute_sample_layer_stats(record["entropy_scores"], DEEP_LAYERS)["mean"]
        grouped[label].append((float(shallow), float(deep)))

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for label, rows in grouped.items():
        if not rows:
            result[label] = (np.array([], dtype=np.float32), np.array([], dtype=np.float32))
            continue
        array = np.asarray(rows, dtype=np.float32)
        result[label] = (array[:, 0], array[:, 1])
    return result


def plot_layerwise_entropy_curve(merged_records: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Plot mean layerwise entropy curves for correct vs hallucinated samples."""
    plot = _require_matplotlib()
    grouped = _group_layer_means(merged_records)
    layers = np.arange(28, dtype=np.int32)

    plot.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plot.subplots(figsize=(9, 5.5))
    for label in (0, 1):
        layer_matrix = grouped[label]
        if layer_matrix.size == 0:
            continue
        mean_curve = layer_matrix.mean(axis=0)
        std_curve = layer_matrix.std(axis=0, ddof=0)
        color = LABEL_COLORS[label]
        ax.plot(layers, mean_curve, color=color, linewidth=2.2, label=LABEL_NAMES[label])
        ax.fill_between(layers, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.18)

    ax.axvline(9, color="#555555", linestyle="--", linewidth=1.2)
    ax.axvline(19, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_title("Layerwise Entropy Trajectories")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Entropy")
    ax.set_xlim(0, 27)
    ax.legend(frameon=True)
    fig.tight_layout()

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plot.close(fig)


def plot_shallow_vs_deep_scatter(merged_records: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Plot shallow vs deep mean entropy for each sample."""
    plot = _require_matplotlib()
    grouped = _scatter_points(merged_records)

    plot.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plot.subplots(figsize=(7, 6))

    all_x = []
    all_y = []
    for label in (0, 1):
        shallow, deep = grouped[label]
        if shallow.size == 0:
            continue
        all_x.append(shallow)
        all_y.append(deep)
        ax.scatter(
            shallow,
            deep,
            s=26,
            alpha=0.45,
            color=LABEL_COLORS[label],
            edgecolors="none",
            label=LABEL_NAMES[label],
        )

    if all_x and all_y:
        min_value = float(min(np.min(values) for values in all_x + all_y))
        max_value = float(max(np.max(values) for values in all_x + all_y))
    else:
        min_value, max_value = 0.0, 1.0
    ax.plot([min_value, max_value], [min_value, max_value], color="#444444", linestyle="--", linewidth=1.2)

    ax.set_title("Shallow vs Deep Mean Entropy")
    ax.set_xlabel("Shallow Mean Entropy (layers 0-9)")
    ax.set_ylabel("Deep Mean Entropy (layers 19-27)")
    ax.legend(frameon=True)
    fig.tight_layout()

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plot.close(fig)


def plot_mismatch_distribution(merged_records: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Plot the distribution of deep-minus-shallow entropy by label."""
    plot = _require_matplotlib()
    grouped = _scatter_points(merged_records)

    mismatch_values: dict[int, np.ndarray] = {}
    for label in (0, 1):
        shallow, deep = grouped[label]
        mismatch_values[label] = deep - shallow

    all_values = [values for values in mismatch_values.values() if values.size]
    if all_values:
        min_value = float(min(values.min() for values in all_values))
        max_value = float(max(values.max() for values in all_values))
        bins = np.linspace(min_value, max_value, 35)
    else:
        bins = 30

    plot.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plot.subplots(figsize=(8, 5.5))
    for label in (0, 1):
        values = mismatch_values[label]
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            alpha=0.5,
            color=LABEL_COLORS[label],
            label=LABEL_NAMES[label],
            density=False,
        )

    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_title("Distribution of Deep-Shallow Entropy Mismatch")
    ax.set_xlabel("Deep Mean Entropy - Shallow Mean Entropy")
    ax.set_ylabel("Count")
    ax.legend(frameon=True)
    fig.tight_layout()

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plot.close(fig)


__all__ = [
    "plot_layerwise_entropy_curve",
    "plot_shallow_vs_deep_scatter",
    "plot_mismatch_distribution",
]
