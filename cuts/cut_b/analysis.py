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
    from scipy import stats as scipy_stats
except ModuleNotFoundError:
    scipy_stats = None


SHALLOW_LAYERS = slice(0, 10)
DEEP_LAYERS = slice(19, 28)
ALL_LAYERS = slice(0, 28)
LABEL_NAMES = {0: "correct", 1: "hallucinated"}


def _require_scipy_stats():
    if scipy_stats is None:
        raise RuntimeError("Missing dependency `scipy`. Install it with `pip install scipy`.")
    return scipy_stats


def _as_entropy_matrix(entropy_scores: Any) -> np.ndarray:
    matrix = np.asarray(entropy_scores, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected entropy scores with shape [layers, tokens], got {matrix.shape}.")
    if matrix.shape[0] < 28:
        raise ValueError(f"Expected at least 28 entropy layers, got {matrix.shape[0]}.")
    return matrix


def _layer_mean_vector(entropy_scores: Any) -> np.ndarray:
    matrix = _as_entropy_matrix(entropy_scores)
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return matrix.mean(axis=1).astype(np.float32)


def _safe_number(value: Any) -> float | None:
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
        "mean": _safe_number(array.mean()),
        "std": _safe_number(array.std(ddof=0)),
        "median": _safe_number(np.median(array)),
        "min": _safe_number(array.min()),
        "max": _safe_number(array.max()),
        "q25": _safe_number(np.percentile(array, 25)),
        "q75": _safe_number(np.percentile(array, 75)),
    }


def _cohens_d_independent(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size < 2 or y_array.size < 2:
        return None

    x_var = x_array.var(ddof=1)
    y_var = y_array.var(ddof=1)
    dof = x_array.size + y_array.size - 2
    if dof <= 0:
        return None

    pooled_var = (((x_array.size - 1) * x_var) + ((y_array.size - 1) * y_var)) / dof
    if pooled_var <= 0:
        mean_diff = x_array.mean() - y_array.mean()
        return 0.0 if np.isclose(mean_diff, 0.0) else None
    return _safe_number((x_array.mean() - y_array.mean()) / np.sqrt(pooled_var))


def _cohens_d_paired(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size != y_array.size or x_array.size < 2:
        return None

    diff = x_array - y_array
    diff_std = diff.std(ddof=1)
    if diff_std <= 0:
        mean_diff = diff.mean()
        return 0.0 if np.isclose(mean_diff, 0.0) else None
    return _safe_number(diff.mean() / diff_std)


def _run_mann_whitney(x: Sequence[float], y: Sequence[float], label: str) -> dict[str, Any]:
    stats = _require_scipy_stats()
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)

    payload: dict[str, Any] = {
        "test": "mannwhitneyu",
        "label": label,
        "comparison": {"x": "correct", "y": "hallucinated"},
        "n_x": int(x_array.size),
        "n_y": int(y_array.size),
        "effect_size_cohens_d": _cohens_d_independent(x_array, y_array),
        "x_summary": _summarize_values(x_array),
        "y_summary": _summarize_values(y_array),
    }
    if x_array.size == 0 or y_array.size == 0:
        payload["statistic"] = None
        payload["p_value"] = None
        payload["error"] = "Both groups must contain at least one sample."
        return payload

    result = stats.mannwhitneyu(x_array, y_array, alternative="two-sided")
    payload["statistic"] = _safe_number(result.statistic)
    payload["p_value"] = _safe_number(result.pvalue)
    return payload


def _run_wilcoxon(x: Sequence[float], y: Sequence[float], label: str, group_name: str) -> dict[str, Any]:
    stats = _require_scipy_stats()
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)

    payload: dict[str, Any] = {
        "test": "wilcoxon",
        "label": label,
        "comparison": {"x": "shallow", "y": "deep"},
        "group": group_name,
        "n_pairs": int(min(x_array.size, y_array.size)),
        "effect_size_cohens_d": _cohens_d_paired(x_array, y_array),
        "x_summary": _summarize_values(x_array),
        "y_summary": _summarize_values(y_array),
    }
    if x_array.size != y_array.size or x_array.size == 0:
        payload["statistic"] = None
        payload["p_value"] = None
        payload["error"] = "Wilcoxon requires equal-length paired inputs."
        return payload

    try:
        result = stats.wilcoxon(x_array, y_array, alternative="two-sided", zero_method="zsplit")
    except ValueError as exc:
        payload["statistic"] = None
        payload["p_value"] = None
        payload["error"] = str(exc)
        return payload

    payload["statistic"] = _safe_number(result.statistic)
    payload["p_value"] = _safe_number(result.pvalue)
    return payload


def compute_sample_layer_stats(entropy_scores: Any, layer_range: slice) -> dict[str, Any]:
    """Summarize one sample's entropy over a layer range after token averaging."""
    layer_means = _layer_mean_vector(entropy_scores)[layer_range]
    if layer_means.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n_layers": 0,
            "n_tokens": int(_as_entropy_matrix(entropy_scores).shape[1]),
            "layer_means": [],
        }

    return {
        "mean": float(layer_means.mean()),
        "std": float(layer_means.std(ddof=0)),
        "min": float(layer_means.min()),
        "max": float(layer_means.max()),
        "n_layers": int(layer_means.size),
        "n_tokens": int(_as_entropy_matrix(entropy_scores).shape[1]),
        "layer_means": [float(value) for value in layer_means.tolist()],
    }


def analyze_shallow_deep_confidence(merged_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Analyze shallow-vs-deep entropy patterns for correct and hallucinated samples."""
    _require_scipy_stats()

    sample_rows: list[dict[str, Any]] = []
    for record in merged_records:
        label = int(record["label"])
        shallow_stats = compute_sample_layer_stats(record["entropy_scores"], SHALLOW_LAYERS)
        deep_stats = compute_sample_layer_stats(record["entropy_scores"], DEEP_LAYERS)
        overall_stats = compute_sample_layer_stats(record["entropy_scores"], ALL_LAYERS)

        sample_rows.append(
            {
                "index": int(record["index"]),
                "candidate_index": int(record.get("candidate_index", 0)),
                "label": label,
                "label_name": LABEL_NAMES.get(label, str(label)),
                "shallow_entropy_mean": shallow_stats["mean"],
                "deep_entropy_mean": deep_stats["mean"],
                "overall_entropy_mean": overall_stats["mean"],
                "deep_minus_shallow": deep_stats["mean"] - shallow_stats["mean"],
            }
        )

    grouped = {
        0: [row for row in sample_rows if row["label"] == 0],
        1: [row for row in sample_rows if row["label"] == 1],
    }

    correct_shallow = np.asarray([row["shallow_entropy_mean"] for row in grouped[0]], dtype=np.float64)
    hallucinated_shallow = np.asarray([row["shallow_entropy_mean"] for row in grouped[1]], dtype=np.float64)
    correct_deep = np.asarray([row["deep_entropy_mean"] for row in grouped[0]], dtype=np.float64)
    hallucinated_deep = np.asarray([row["deep_entropy_mean"] for row in grouped[1]], dtype=np.float64)
    correct_overall = np.asarray([row["overall_entropy_mean"] for row in grouped[0]], dtype=np.float64)
    hallucinated_overall = np.asarray([row["overall_entropy_mean"] for row in grouped[1]], dtype=np.float64)
    correct_mismatch = np.asarray([row["deep_minus_shallow"] for row in grouped[0]], dtype=np.float64)
    hallucinated_mismatch = np.asarray([row["deep_minus_shallow"] for row in grouped[1]], dtype=np.float64)

    results = {
        "hypothesis": (
            "Hallucinated responses lock in high confidence (low entropy) in shallow layers "
            "while deeper layers remain more uncertain than correct responses."
        ),
        "layer_ranges": {
            "shallow": {"start": 0, "end_exclusive": 10, "description": "layers 0-9"},
            "deep": {"start": 19, "end_exclusive": 28, "description": "layers 19-27"},
            "overall": {"start": 0, "end_exclusive": 28, "description": "layers 0-27"},
        },
        "n_samples": len(sample_rows),
        "n_correct": len(grouped[0]),
        "n_hallucinated": len(grouped[1]),
        "group_summaries": {
            "correct": {
                "shallow_entropy_mean": _summarize_values(correct_shallow),
                "deep_entropy_mean": _summarize_values(correct_deep),
                "overall_entropy_mean": _summarize_values(correct_overall),
                "deep_minus_shallow": _summarize_values(correct_mismatch),
            },
            "hallucinated": {
                "shallow_entropy_mean": _summarize_values(hallucinated_shallow),
                "deep_entropy_mean": _summarize_values(hallucinated_deep),
                "overall_entropy_mean": _summarize_values(hallucinated_overall),
                "deep_minus_shallow": _summarize_values(hallucinated_mismatch),
            },
        },
        "tests": {
            "shallow_correct_vs_hallucinated": _run_mann_whitney(
                correct_shallow,
                hallucinated_shallow,
                label="shallow entropy: correct vs hallucinated",
            ),
            "deep_correct_vs_hallucinated": _run_mann_whitney(
                correct_deep,
                hallucinated_deep,
                label="deep entropy: correct vs hallucinated",
            ),
            "hallucinated_shallow_vs_deep": _run_wilcoxon(
                hallucinated_shallow,
                hallucinated_deep,
                label="hallucinated shallow vs deep entropy",
                group_name="hallucinated",
            ),
            "correct_shallow_vs_deep": _run_wilcoxon(
                correct_shallow,
                correct_deep,
                label="correct shallow vs deep entropy",
                group_name="correct",
            ),
        },
    }
    return results


def _format_summary_line(name: str, summary: dict[str, Any]) -> str:
    return (
        f"{name}: n={summary['n']} "
        f"mean={summary['mean']:.4f} "
        f"std={summary['std']:.4f} "
        f"median={summary['median']:.4f}"
        if summary["n"]
        else f"{name}: n=0"
    )


def _format_test_line(name: str, payload: dict[str, Any]) -> str:
    statistic = payload.get("statistic")
    p_value = payload.get("p_value")
    effect = payload.get("effect_size_cohens_d")
    error = payload.get("error")
    if statistic is None or p_value is None:
        suffix = f"unavailable ({error})" if error else "unavailable"
        return f"{name}: {suffix}"
    effect_text = "n/a" if effect is None else f"{effect:.4f}"
    return f"{name}: statistic={statistic:.4f} p={p_value:.4e} Cohen's d={effect_text}"


def format_analysis_report(results: dict[str, Any]) -> str:
    """Render a human-readable report for the Cut B statistical analysis."""
    lines = [
        "Cut B: Shallow Overconfidence Hypothesis",
        "=" * 44,
        "",
        results["hypothesis"],
        "",
        f"Samples: total={results['n_samples']} correct={results['n_correct']} hallucinated={results['n_hallucinated']}",
        "Layer ranges:",
        f"  shallow: {results['layer_ranges']['shallow']['description']}",
        f"  deep:    {results['layer_ranges']['deep']['description']}",
        "",
        "Group summaries:",
        f"  Correct shallow: {_format_summary_line('entropy', results['group_summaries']['correct']['shallow_entropy_mean'])}",
        f"  Correct deep:    {_format_summary_line('entropy', results['group_summaries']['correct']['deep_entropy_mean'])}",
        f"  Correct overall: {_format_summary_line('entropy', results['group_summaries']['correct']['overall_entropy_mean'])}",
        (
            "  Correct mismatch (deep-shallow): "
            f"{_format_summary_line('delta', results['group_summaries']['correct']['deep_minus_shallow'])}"
        ),
        f"  Hallucinated shallow: {_format_summary_line('entropy', results['group_summaries']['hallucinated']['shallow_entropy_mean'])}",
        f"  Hallucinated deep:    {_format_summary_line('entropy', results['group_summaries']['hallucinated']['deep_entropy_mean'])}",
        f"  Hallucinated overall: {_format_summary_line('entropy', results['group_summaries']['hallucinated']['overall_entropy_mean'])}",
        (
            "  Hallucinated mismatch (deep-shallow): "
            f"{_format_summary_line('delta', results['group_summaries']['hallucinated']['deep_minus_shallow'])}"
        ),
        "",
        "Statistical tests:",
        f"  {_format_test_line('Mann-Whitney U (shallow, correct vs hallucinated)', results['tests']['shallow_correct_vs_hallucinated'])}",
        f"  {_format_test_line('Mann-Whitney U (deep, correct vs hallucinated)', results['tests']['deep_correct_vs_hallucinated'])}",
        f"  {_format_test_line('Wilcoxon (hallucinated, shallow vs deep)', results['tests']['hallucinated_shallow_vs_deep'])}",
        f"  {_format_test_line('Wilcoxon (correct, shallow vs deep)', results['tests']['correct_shallow_vs_deep'])}",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "SHALLOW_LAYERS",
    "DEEP_LAYERS",
    "ALL_LAYERS",
    "compute_sample_layer_stats",
    "analyze_shallow_deep_confidence",
    "format_analysis_report",
]
