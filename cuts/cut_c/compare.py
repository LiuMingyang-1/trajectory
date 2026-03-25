"""Post-training analysis for Cut C gate behavior."""

from __future__ import annotations

import json
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

from cut_c.gating import GATE_ENTROPY_LAYERS
from shared.data_loader import load_combined_span_dataset
from shared.eval_utils import dump_json, evaluate_binary_predictions, read_jsonl


LABEL_NAMES = {0: "Correct", 1: "Hallucinated"}
LABEL_COLORS = {0: "#2166ac", 1: "#b2182b"}
GATE_THRESHOLD = 0.5


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
        }
    return {
        "n": int(array.size),
        "mean": _safe_float(array.mean()),
        "std": _safe_float(array.std(ddof=0)),
        "median": _safe_float(np.median(array)),
        "min": _safe_float(array.min()),
        "max": _safe_float(array.max()),
        "q25": _safe_float(np.percentile(array, 25)),
        "q75": _safe_float(np.percentile(array, 75)),
    }


def _histogram_payload(values: Sequence[float], bins: int = 12) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"bin_edges": [], "counts": []}
    counts, edges = np.histogram(array, bins=bins, range=(0.0, 1.0))
    return {
        "bin_edges": [float(value) for value in edges.tolist()],
        "counts": [int(value) for value in counts.tolist()],
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size < 2 or y_array.size < 2:
        return None
    x_std = x_array.std(ddof=0)
    y_std = y_array.std(ddof=0)
    if x_std <= 0 or y_std <= 0:
        return None
    return _safe_float(np.corrcoef(x_array, y_array)[0, 1])


def _spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size < 2 or y_array.size < 2:
        return None
    return _pearson_correlation(_average_ranks(x_array), _average_ranks(y_array))


def _linear_fit(x: Sequence[float], y: Sequence[float]) -> dict[str, float | None]:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size < 2 or np.allclose(x_array, x_array[0]):
        return {"slope": None, "intercept": None}
    slope, intercept = np.polyfit(x_array, y_array, deg=1)
    return {"slope": _safe_float(slope), "intercept": _safe_float(intercept)}


def _ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _metrics_path_for_prediction(prediction_path: Path) -> Path:
    return prediction_path.with_name(prediction_path.name.replace(".oof_predictions.jsonl", ".metrics.json"))


def _load_metrics(prediction_path: Path) -> dict[str, Any]:
    metrics_path = _metrics_path_for_prediction(prediction_path)
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _align_prediction_rows(
    dataset_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(dataset_rows) != len(prediction_rows):
        raise ValueError(
            f"Dataset rows ({len(dataset_rows)}) and prediction rows ({len(prediction_rows)}) must match."
        )

    aligned: list[dict[str, Any]] = []
    for index, (dataset_row, prediction_row) in enumerate(zip(dataset_rows, prediction_rows)):
        dataset_sample_id = dataset_row.get("sample_id")
        prediction_sample_id = prediction_row.get("sample_id")
        if dataset_sample_id != prediction_sample_id:
            raise ValueError(
                f"Sample mismatch at row {index}: dataset={dataset_sample_id!r} prediction={prediction_sample_id!r}"
            )

        dataset_span_id = dataset_row.get("span_id")
        prediction_span_id = prediction_row.get("span_id")
        if dataset_span_id != prediction_span_id:
            raise ValueError(
                f"Span mismatch at row {index}: dataset={dataset_span_id!r} prediction={prediction_span_id!r}"
            )

        entropy_vector = np.asarray(dataset_row["entropy_vector"], dtype=np.float32)
        mid_entropy = entropy_vector[GATE_ENTROPY_LAYERS]
        aligned.append(
            {
                **prediction_row,
                "sample_label": int(dataset_row["sample_label"]),
                "silver_label_int": -1 if dataset_row.get("silver_label") is None else int(dataset_row["silver_label"]),
                "entropy_mid_mean": float(mid_entropy.mean()) if mid_entropy.size else 0.0,
                "entropy_mid_max": float(mid_entropy.max()) if mid_entropy.size else 0.0,
            }
        )
    return aligned


def _gate_distribution_by_label(rows: Sequence[dict[str, Any]], label_key: str) -> dict[str, Any]:
    grouped: dict[int, np.ndarray] = {}
    for label in (0, 1):
        values = [
            float(row["gate"])
            for row in rows
            if row.get("gate") is not None and int(row[label_key]) == label
        ]
        grouped[label] = np.asarray(values, dtype=np.float64)

    result = {
        "label_key": label_key,
        "threshold": GATE_THRESHOLD,
        "groups": {},
        "mean_difference_hallucinated_minus_correct": None,
        "high_gate_rate_difference_hallucinated_minus_correct": None,
    }
    for label in (0, 1):
        values = grouped[label]
        high_gate_rate = _safe_float((values > GATE_THRESHOLD).mean()) if values.size else None
        result["groups"][LABEL_NAMES[label].lower()] = {
            "label": label,
            "summary": _summarize_values(values),
            "high_gate_rate": high_gate_rate,
            "histogram": _histogram_payload(values),
        }

    correct_mean = result["groups"]["correct"]["summary"]["mean"]
    hallucinated_mean = result["groups"]["hallucinated"]["summary"]["mean"]
    correct_high = result["groups"]["correct"]["high_gate_rate"]
    hallucinated_high = result["groups"]["hallucinated"]["high_gate_rate"]
    if correct_mean is not None and hallucinated_mean is not None:
        result["mean_difference_hallucinated_minus_correct"] = _safe_float(hallucinated_mean - correct_mean)
    if correct_high is not None and hallucinated_high is not None:
        result["high_gate_rate_difference_hallucinated_minus_correct"] = _safe_float(
            hallucinated_high - correct_high
        )
    return result


def _entropy_bin_summary(
    entropy_values: np.ndarray,
    gate_values: np.ndarray,
    sample_labels: np.ndarray,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    if entropy_values.size == 0:
        return []

    quantiles = np.quantile(entropy_values, np.linspace(0.0, 1.0, n_bins + 1))
    if np.unique(quantiles).size <= 2:
        quantiles = np.linspace(float(entropy_values.min()), float(entropy_values.max()), n_bins + 1)
    if np.unique(quantiles).size <= 1:
        quantiles = np.array([float(entropy_values.min()), float(entropy_values.max()) + 1e-6], dtype=np.float64)

    bins: list[dict[str, Any]] = []
    for index in range(len(quantiles) - 1):
        left = float(quantiles[index])
        right = float(quantiles[index + 1])
        if index == len(quantiles) - 2:
            mask = (entropy_values >= left) & (entropy_values <= right)
        else:
            mask = (entropy_values >= left) & (entropy_values < right)
        if not mask.any():
            continue
        bins.append(
            {
                "bin_index": index,
                "left": left,
                "right": right,
                "n": int(mask.sum()),
                "entropy_mean": _safe_float(entropy_values[mask].mean()),
                "gate_mean": _safe_float(gate_values[mask].mean()),
                "gate_std": _safe_float(gate_values[mask].std(ddof=0)),
                "hallucinated_rate": _safe_float(sample_labels[mask].mean()),
            }
        )
    return bins


def _gate_entropy_relationship(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    usable_rows = [row for row in rows if row.get("gate") is not None]
    gate_values = np.asarray([float(row["gate"]) for row in usable_rows], dtype=np.float64)
    entropy_values = np.asarray([float(row["entropy_mid_mean"]) for row in usable_rows], dtype=np.float64)
    sample_labels = np.asarray([int(row["sample_label"]) for row in usable_rows], dtype=np.int32)

    result = {
        "overall": {
            "n": int(gate_values.size),
            "gate_summary": _summarize_values(gate_values),
            "entropy_mid_summary": _summarize_values(entropy_values),
            "pearson_correlation": _pearson_correlation(gate_values, entropy_values),
            "spearman_correlation": _spearman_correlation(gate_values, entropy_values),
            "linear_fit": _linear_fit(entropy_values, gate_values),
        },
        "by_sample_label": {},
        "entropy_bins": _entropy_bin_summary(entropy_values, gate_values, sample_labels),
    }

    for label in (0, 1):
        mask = sample_labels == label
        result["by_sample_label"][LABEL_NAMES[label].lower()] = {
            "n": int(mask.sum()),
            "gate_summary": _summarize_values(gate_values[mask]),
            "entropy_mid_summary": _summarize_values(entropy_values[mask]),
            "pearson_correlation": _pearson_correlation(gate_values[mask], entropy_values[mask]),
        }
    return result


def _subgroup_performance(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labeled_rows = [
        row
        for row in rows
        if row.get("gate") is not None and int(row.get("silver_label_int", -1)) in {0, 1} and row.get("probability") is not None
    ]
    if not labeled_rows:
        return {
            "threshold": GATE_THRESHOLD,
            "overall": {"n_rows": 0},
            "groups": {},
        }

    gate_values = np.asarray([float(row["gate"]) for row in labeled_rows], dtype=np.float64)
    labels = np.asarray([int(row["silver_label_int"]) for row in labeled_rows], dtype=np.int32)
    probabilities = np.asarray([float(row["probability"]) for row in labeled_rows], dtype=np.float64)
    change_probabilities = np.asarray(
        [float(row["change_probability"]) for row in labeled_rows],
        dtype=np.float64,
    )
    icr_probabilities = np.asarray(
        [float(row["icr_probability"]) for row in labeled_rows],
        dtype=np.float64,
    )

    groups = {
        "gate_le_0_5": gate_values <= GATE_THRESHOLD,
        "gate_gt_0_5": gate_values > GATE_THRESHOLD,
    }

    payload = {
        "threshold": GATE_THRESHOLD,
        "overall": {
            "n_rows": int(labels.size),
            "positive_rate": _safe_float(labels.mean()),
            "gate_summary": _summarize_values(gate_values),
        },
        "groups": {},
    }
    for group_name, mask in groups.items():
        group_labels = labels[mask]
        group_probs = probabilities[mask]
        group_gate = gate_values[mask]
        group_change = change_probabilities[mask]
        group_icr = icr_probabilities[mask]

        payload["groups"][group_name] = {
            "n_rows": int(mask.sum()),
            "positive_rate": _safe_float(group_labels.mean()) if group_labels.size else None,
            "gate_summary": _summarize_values(group_gate),
            "probability_summary": _summarize_values(group_probs),
            "metrics": evaluate_binary_predictions(group_labels, group_probs) if group_labels.size else {},
            "change_expert_metrics": evaluate_binary_predictions(group_labels, group_change) if group_labels.size else {},
            "icr_expert_metrics": evaluate_binary_predictions(group_labels, group_icr) if group_labels.size else {},
        }
    return payload


def _plot_gate_distribution(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return

    usable = [
        row
        for row in rows
        if row.get("gate") is not None and int(row.get("sample_label", -1)) in {0, 1}
    ]
    if not usable:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.linspace(0.0, 1.0, 25)

    for label in (0, 1):
        values = np.asarray(
            [float(row["gate"]) for row in usable if int(row["sample_label"]) == label],
            dtype=np.float64,
        )
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

    ax.axvline(GATE_THRESHOLD, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_title("Gate Distribution by Sample Label")
    ax.set_xlabel("Gate Value")
    ax.set_ylabel("Count")
    ax.legend(frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gate_vs_entropy(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return

    usable = [row for row in rows if row.get("gate") is not None]
    if not usable:
        return

    x = np.asarray([float(row["entropy_mid_mean"]) for row in usable], dtype=np.float64)
    y = np.asarray([float(row["gate"]) for row in usable], dtype=np.float64)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for label in (0, 1):
        mask = np.asarray([int(row["sample_label"]) == label for row in usable], dtype=bool)
        if not mask.any():
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=22,
            alpha=0.35,
            color=LABEL_COLORS[label],
            edgecolors="none",
            label=LABEL_NAMES[label],
        )

    fit = _linear_fit(x, y)
    if fit["slope"] is not None and fit["intercept"] is not None:
        x_line = np.linspace(float(x.min()), float(x.max()), 200)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(x_line, y_line, color="#222222", linewidth=1.8, linestyle="-")

    ax.axhline(GATE_THRESHOLD, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_title("Gate vs Middle-Layer Entropy")
    ax.set_xlabel("Mean Entropy (layers 13-17)")
    ax.set_ylabel("Gate Value")
    ax.legend(frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_subgroup_performance(summary: dict[str, Any], output_path: Path) -> None:
    if plt is None:
        return

    groups = summary.get("groups", {})
    labels = ["gate<=0.5", "gate>0.5"]
    metrics = ["AUROC", "AUPRC", "F1"]
    values = []
    for key in ("gate_le_0_5", "gate_gt_0_5"):
        group_metrics = groups.get(key, {}).get("metrics", {})
        values.append([group_metrics.get(metric, 0.0) for metric in metrics])

    if not values:
        return

    data = np.asarray(values, dtype=np.float64)
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.22

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = ["#2166ac", "#4d9221", "#b2182b"]
    for index, metric in enumerate(metrics):
        ax.bar(x + (index - 1) * width, data[:, index], width=width, color=colors[index], label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Metric Value")
    ax.set_title("Span-Level Performance by Gate Subgroup")
    ax.legend(frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_gated_predictions(
    combined_dataset_path: Path,
    prediction_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Analyze one gated prediction file against the combined dataset."""
    dataset_rows = load_combined_span_dataset(Path(combined_dataset_path))[0]
    prediction_rows = read_jsonl(Path(prediction_path))
    if not prediction_rows:
        raise ValueError(f"No prediction rows found in {prediction_path}")
    if "gate" not in prediction_rows[0]:
        raise ValueError(f"Prediction file does not contain gate values: {prediction_path}")

    metrics = _load_metrics(Path(prediction_path))
    feature_set = (
        metrics.get("feature_set")
        or prediction_rows[0].get("feature_set")
        or Path(prediction_path).parent.parent.name
    )
    model_name = metrics.get("model") or prediction_rows[0].get("model") or Path(prediction_path).stem

    aligned_rows = _align_prediction_rows(dataset_rows, prediction_rows)
    model_output_dir = Path(output_dir)
    figures_dir = model_output_dir / "figures"
    _ensure_dir(figures_dir)

    gate_distribution_sample = _gate_distribution_by_label(aligned_rows, label_key="sample_label")
    gate_distribution_silver = _gate_distribution_by_label(
        [row for row in aligned_rows if int(row["silver_label_int"]) in {0, 1}],
        label_key="silver_label_int",
    )
    gate_entropy = _gate_entropy_relationship(aligned_rows)
    subgroup = _subgroup_performance(aligned_rows)

    figure_paths: dict[str, str] = {}
    if plt is not None:
        figure_paths = {
            "gate_distribution_by_sample_label": str(figures_dir / "gate_distribution_by_sample_label.png"),
            "gate_vs_entropy": str(figures_dir / "gate_vs_entropy.png"),
            "gate_subgroup_performance": str(figures_dir / "gate_subgroup_performance.png"),
        }
        _plot_gate_distribution(aligned_rows, Path(figure_paths["gate_distribution_by_sample_label"]))
        _plot_gate_vs_entropy(aligned_rows, Path(figure_paths["gate_vs_entropy"]))
        _plot_subgroup_performance(subgroup, Path(figure_paths["gate_subgroup_performance"]))

    summary = {
        "combined_dataset_path": str(combined_dataset_path),
        "prediction_path": str(prediction_path),
        "metrics_path": str(_metrics_path_for_prediction(Path(prediction_path))),
        "feature_set": feature_set,
        "model": model_name,
        "matplotlib_available": plt is not None,
        "overall_sample_auroc": metrics.get("sample_level", {}).get("max", {}).get("AUROC_mean"),
        "overall_span_auroc": metrics.get("span_level", {}).get("AUROC_mean"),
        "n_rows": len(aligned_rows),
        "n_labeled_rows": int(sum(int(row["silver_label_int"]) in {0, 1} for row in aligned_rows)),
        "gate_distribution_by_sample_label": gate_distribution_sample,
        "gate_distribution_by_silver_label": gate_distribution_silver,
        "gate_entropy_relationship": gate_entropy,
        "subgroup_performance": subgroup,
        "figure_paths": figure_paths,
    }
    dump_json(model_output_dir / "gate_analysis_summary.json", summary)
    return summary


def run_gate_comparison(
    combined_dataset_path: Path,
    training_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run gate behavior analysis for all gated models under a training directory."""
    training_dir = Path(training_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_paths = sorted(training_dir.rglob("*.oof_predictions.jsonl"))
    model_summaries: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        metrics = _load_metrics(prediction_path)
        feature_set = metrics.get("feature_set")
        if feature_set is None:
            rows = read_jsonl(prediction_path)
            if not rows:
                continue
            feature_set = rows[0].get("feature_set")
        if not str(feature_set).startswith("gated_"):
            continue

        summary = analyze_gated_predictions(
            combined_dataset_path=combined_dataset_path,
            prediction_path=prediction_path,
            output_dir=output_dir / str(feature_set),
        )
        model_summaries.append(summary)

    if not model_summaries:
        raise FileNotFoundError(f"No gated `.oof_predictions.jsonl` files found under {training_dir}.")

    primary_model = max(
        model_summaries,
        key=lambda item: (
            float("-inf") if item.get("overall_sample_auroc") is None else float(item["overall_sample_auroc"]),
            float("-inf") if item.get("overall_span_auroc") is None else float(item["overall_span_auroc"]),
        ),
    )
    summary = {
        "combined_dataset_path": str(combined_dataset_path),
        "training_dir": str(training_dir),
        "output_dir": str(output_dir),
        "matplotlib_available": plt is not None,
        "n_models": len(model_summaries),
        "model_summaries": model_summaries,
        "primary_model": {
            "feature_set": primary_model["feature_set"],
            "model": primary_model["model"],
            "overall_sample_auroc": primary_model.get("overall_sample_auroc"),
            "overall_span_auroc": primary_model.get("overall_span_auroc"),
            "summary_path": str(output_dir / primary_model["feature_set"] / "gate_analysis_summary.json"),
        },
    }
    dump_json(output_dir / "comparison_summary.json", summary)
    return summary


__all__ = ["analyze_gated_predictions", "run_gate_comparison"]
