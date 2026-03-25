"""Error analysis for Cut A confidence trajectory models."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ensure_spanlab_importable


ensure_spanlab_importable()

from shared.data_loader import load_entropy_records, load_icr_records, merge_icr_entropy
from shared.eval_utils import aggregate_sample_predictions, dump_json, read_jsonl
from spanlab.alignment import build_sample_id


SHALLOW_LAYERS = slice(0, 10)
MID_LAYERS = slice(10, 19)
DEEP_LAYERS = slice(19, 28)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _metric_value(metrics: dict[str, Any], section: str, key: str, mode: str | None = None) -> float | None:
    if section == "sample_level":
        value = metrics.get(section, {}).get(mode or "max", {}).get(key)
    else:
        value = metrics.get(section, {}).get(key)
    return _safe_float(value)


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
        }
    return {
        "n": int(array.size),
        "mean": _safe_float(array.mean()),
        "std": _safe_float(array.std(ddof=0)),
        "median": _safe_float(np.median(array)),
        "min": _safe_float(array.min()),
        "max": _safe_float(array.max()),
    }


def _layer_mean_vector(entropy_scores: Any) -> np.ndarray:
    matrix = np.asarray(entropy_scores, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected entropy scores with shape [layers, tokens], got {matrix.shape}.")
    if matrix.shape[0] < 28:
        raise ValueError(f"Expected at least 28 entropy layers, got {matrix.shape[0]}.")
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return matrix.mean(axis=1).astype(np.float32)


def _linear_slope(vector: np.ndarray) -> float:
    layers = np.arange(vector.shape[0], dtype=np.float32)
    centered = layers - layers.mean()
    denom = float((centered**2).sum())
    if denom <= 0:
        return 0.0
    return float((vector * centered).sum() / denom)


def _entropy_features_for_record(record: dict[str, Any]) -> dict[str, float]:
    layer_means = _layer_mean_vector(record["entropy_scores"])
    deltas = np.diff(layer_means)

    shallow_mean = float(layer_means[SHALLOW_LAYERS].mean()) if layer_means[SHALLOW_LAYERS].size else 0.0
    mid_mean = float(layer_means[MID_LAYERS].mean()) if layer_means[MID_LAYERS].size else 0.0
    deep_mean = float(layer_means[DEEP_LAYERS].mean()) if layer_means[DEEP_LAYERS].size else 0.0

    return {
        "shallow_entropy_mean": shallow_mean,
        "mid_entropy_mean": mid_mean,
        "deep_entropy_mean": deep_mean,
        "overall_entropy_mean": float(layer_means.mean()),
        "deep_minus_shallow": deep_mean - shallow_mean,
        "entropy_slope": _linear_slope(layer_means),
        "max_entropy_layer": float(np.argmax(layer_means)),
        "delta_entropy_mean": float(deltas.mean()) if deltas.size else 0.0,
        "delta_entropy_abs_mean": float(np.abs(deltas).mean()) if deltas.size else 0.0,
    }


def _aggregate_rows_to_sample_predictions(rows: Sequence[dict[str, Any]], mode: str = "max") -> dict[str, dict[str, Any]]:
    usable_rows = [row for row in rows if row.get("probability") is not None]
    if not usable_rows:
        return {}

    probabilities = np.asarray([float(row["probability"]) for row in usable_rows], dtype=np.float32)
    aggregated = aggregate_sample_predictions(usable_rows, probabilities, top_k=3)[mode]
    span_counts = Counter(str(row["sample_id"]) for row in usable_rows)

    sample_predictions: dict[str, dict[str, Any]] = {}
    for sample_id, label, prob in zip(
        aggregated["sample_ids"],
        aggregated["labels"],
        aggregated["probs"],
    ):
        sample_id_str = str(sample_id)
        sample_predictions[sample_id_str] = {
            "sample_id": sample_id_str,
            "sample_label": int(label),
            "probability": float(prob),
            "n_spans": int(span_counts[sample_id_str]),
        }
    return sample_predictions


def _load_metrics_for_prediction(prediction_path: Path) -> dict[str, Any]:
    metrics_path = prediction_path.with_name(prediction_path.name.replace(".oof_predictions.jsonl", ".metrics.json"))
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _prediction_key(root: Path, prediction_path: Path) -> str:
    try:
        relative = prediction_path.relative_to(root)
    except ValueError:
        relative = prediction_path
    return str(relative).replace(".oof_predictions.jsonl", "")


def load_baseline_predictions(baseline_dir: Path) -> dict[str, dict[str, Any]]:
    """Load one or more OOF prediction files and aggregate them to sample level."""
    root = Path(baseline_dir)
    if root.is_file():
        prediction_paths = [root]
        root = root.parent
    else:
        prediction_paths = sorted(root.rglob("*.oof_predictions.jsonl"))

    if not prediction_paths:
        raise FileNotFoundError(f"No `.oof_predictions.jsonl` files found under {baseline_dir}.")

    loaded: dict[str, dict[str, Any]] = {}
    for prediction_path in prediction_paths:
        rows = read_jsonl(prediction_path)
        metrics = _load_metrics_for_prediction(prediction_path)
        model_name = (
            metrics.get("model")
            or (rows[0].get("model") if rows else None)
            or prediction_path.name.replace(".oof_predictions.jsonl", "")
        )
        feature_set = metrics.get("feature_set") or (rows[0].get("feature_set") if rows else None)
        key = _prediction_key(root, prediction_path)

        loaded[key] = {
            "key": key,
            "prediction_path": str(prediction_path),
            "metrics_path": str(
                prediction_path.with_name(prediction_path.name.replace(".oof_predictions.jsonl", ".metrics.json"))
            ),
            "metrics": metrics,
            "family": metrics.get("family") or prediction_path.parent.name,
            "feature_set": feature_set,
            "model": model_name,
            "span_rows": rows,
            "sample_predictions": _aggregate_rows_to_sample_predictions(rows, mode="max"),
            "sample_count": len({row["sample_id"] for row in rows}) if rows else 0,
            "sample_auroc": _metric_value(metrics, "sample_level", "AUROC_mean", mode="max"),
            "span_auroc": _metric_value(metrics, "span_level", "AUROC_mean"),
        }
    return loaded


def _select_primary_prediction(
    prediction_sets: dict[str, dict[str, Any]],
    *,
    prefer_entropy_features: bool,
) -> tuple[str, dict[str, Any]]:
    candidates = list(prediction_sets.items())
    if not candidates:
        raise ValueError("No prediction sets available.")

    if prefer_entropy_features:
        filtered = [
            item
            for item in candidates
            if (item[1].get("feature_set") not in {None, "", "icr_only"})
        ]
        if filtered:
            candidates = filtered

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int]:
        payload = item[1]
        sample_auroc = payload.get("sample_auroc")
        span_auroc = payload.get("span_auroc")
        sample_count = payload.get("sample_count") or 0
        return (
            float("-inf") if sample_auroc is None else float(sample_auroc),
            float("-inf") if span_auroc is None else float(span_auroc),
            int(sample_count),
        )

    return max(candidates, key=sort_key)


def identify_errors(predictions: dict[str, Any], threshold: float = 0.5) -> dict[str, Any]:
    """Split sample-level predictions into TP/TN/FP/FN groups."""
    sample_predictions = predictions.get("sample_predictions", predictions)
    groups = {"tp": [], "tn": [], "fp": [], "fn": []}

    for sample_id, payload in sorted(sample_predictions.items()):
        probability = float(payload["probability"])
        label = int(payload["sample_label"])
        predicted_label = int(probability >= threshold)
        entry = {
            "sample_id": str(sample_id),
            "sample_label": label,
            "predicted_label": predicted_label,
            "probability": probability,
        }
        if predicted_label == 1 and label == 1:
            groups["tp"].append(entry)
        elif predicted_label == 0 and label == 0:
            groups["tn"].append(entry)
        elif predicted_label == 1 and label == 0:
            groups["fp"].append(entry)
        else:
            groups["fn"].append(entry)

    total = sum(len(entries) for entries in groups.values())
    return {
        "threshold": float(threshold),
        "n_samples": total,
        "counts": {name: len(entries) for name, entries in groups.items()},
        "groups": groups,
    }


def compute_error_entropy_stats(
    error_groups: dict[str, Any],
    merged_records: Sequence[dict[str, Any]] | dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute entropy statistics for each error-group sample set."""
    if isinstance(merged_records, dict):
        record_lookup = merged_records
    else:
        record_lookup = {
            build_sample_id(int(record["index"]), int(record.get("candidate_index", 0))): record for record in merged_records
        }

    groups = error_groups.get("groups", error_groups)
    output: dict[str, Any] = {}

    for group_name, entries in groups.items():
        metrics_by_name: dict[str, list[float]] = defaultdict(list)
        missing_sample_ids: list[str] = []

        for entry in entries:
            sample_id = str(entry["sample_id"])
            record = record_lookup.get(sample_id)
            if record is None:
                missing_sample_ids.append(sample_id)
                continue
            for metric_name, metric_value in _entropy_features_for_record(record).items():
                metrics_by_name[metric_name].append(metric_value)

        output[group_name] = {
            "n_requested": len(entries),
            "n_matched": len(entries) - len(missing_sample_ids),
            "missing_sample_ids": missing_sample_ids,
            "metrics": {metric_name: _summarize_values(values) for metric_name, values in metrics_by_name.items()},
        }

    return output


def compare_model_corrections(baseline_preds: dict[str, Any], combined_preds: dict[str, Any]) -> dict[str, Any]:
    """Identify which baseline mistakes are corrected by the combined model."""
    baseline_samples = baseline_preds.get("sample_predictions", baseline_preds)
    combined_samples = combined_preds.get("sample_predictions", combined_preds)

    baseline_ids = set(baseline_samples)
    combined_ids = set(combined_samples)
    shared_ids = sorted(baseline_ids & combined_ids)

    groups = {
        "corrected": [],
        "introduced_errors": [],
        "remaining_errors": [],
        "both_correct": [],
        "corrected_false_positives": [],
        "corrected_false_negatives": [],
        "new_false_positives": [],
        "new_false_negatives": [],
    }

    for sample_id in shared_ids:
        baseline = baseline_samples[sample_id]
        combined = combined_samples[sample_id]

        label = int(baseline["sample_label"])
        baseline_prob = float(baseline["probability"])
        combined_prob = float(combined["probability"])
        baseline_pred = int(baseline_prob >= 0.5)
        combined_pred = int(combined_prob >= 0.5)
        baseline_error = baseline_pred != label
        combined_error = combined_pred != label

        entry = {
            "sample_id": sample_id,
            "sample_label": label,
            "baseline_probability": baseline_prob,
            "combined_probability": combined_prob,
            "baseline_predicted_label": baseline_pred,
            "combined_predicted_label": combined_pred,
        }

        if baseline_error and not combined_error:
            groups["corrected"].append(entry)
            if label == 1:
                groups["corrected_false_negatives"].append(entry)
            else:
                groups["corrected_false_positives"].append(entry)
        elif not baseline_error and combined_error:
            groups["introduced_errors"].append(entry)
            if label == 1:
                groups["new_false_negatives"].append(entry)
            else:
                groups["new_false_positives"].append(entry)
        elif baseline_error and combined_error:
            groups["remaining_errors"].append(entry)
        else:
            groups["both_correct"].append(entry)

    return {
        "threshold": 0.5,
        "n_shared_samples": len(shared_ids),
        "baseline_only_sample_ids": sorted(baseline_ids - combined_ids),
        "combined_only_sample_ids": sorted(combined_ids - baseline_ids),
        "counts": {name: len(entries) for name, entries in groups.items()},
        "groups": groups,
    }


def _candidate_summaries(prediction_sets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, payload in sorted(prediction_sets.items()):
        rows.append(
            {
                "key": key,
                "feature_set": payload.get("feature_set"),
                "family": payload.get("family"),
                "model": payload.get("model"),
                "prediction_path": payload.get("prediction_path"),
                "sample_count": payload.get("sample_count"),
                "sample_auroc": payload.get("sample_auroc"),
                "span_auroc": payload.get("span_auroc"),
            }
        )
    return rows


def run_error_analysis(
    baseline_dir: Path,
    combined_dir: Path,
    entropy_path: Path,
    icr_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run baseline-vs-combined error analysis and save a JSON report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_candidates = load_baseline_predictions(Path(baseline_dir))
    combined_candidates = load_baseline_predictions(Path(combined_dir))

    baseline_key, baseline_model = _select_primary_prediction(
        baseline_candidates,
        prefer_entropy_features=False,
    )
    combined_key, combined_model = _select_primary_prediction(
        combined_candidates,
        prefer_entropy_features=True,
    )

    icr_records = load_icr_records(Path(icr_path))
    entropy_records = load_entropy_records(Path(entropy_path))
    merged_records = merge_icr_entropy(icr_records, entropy_records)

    baseline_errors = identify_errors(baseline_model, threshold=0.5)
    combined_errors = identify_errors(combined_model, threshold=0.5)
    corrections = compare_model_corrections(baseline_model, combined_model)

    baseline_error_entropy = compute_error_entropy_stats(baseline_errors, merged_records)
    combined_error_entropy = compute_error_entropy_stats(combined_errors, merged_records)
    correction_entropy = compute_error_entropy_stats(corrections, merged_records)

    report = {
        "baseline_dir": str(baseline_dir),
        "combined_dir": str(combined_dir),
        "entropy_path": str(entropy_path),
        "icr_path": str(icr_path),
        "candidate_models": {
            "baseline": _candidate_summaries(baseline_candidates),
            "combined": _candidate_summaries(combined_candidates),
        },
        "selected_models": {
            "baseline": {
                "key": baseline_key,
                "feature_set": baseline_model.get("feature_set"),
                "family": baseline_model.get("family"),
                "model": baseline_model.get("model"),
                "prediction_path": baseline_model.get("prediction_path"),
                "sample_auroc": baseline_model.get("sample_auroc"),
                "span_auroc": baseline_model.get("span_auroc"),
            },
            "combined": {
                "key": combined_key,
                "feature_set": combined_model.get("feature_set"),
                "family": combined_model.get("family"),
                "model": combined_model.get("model"),
                "prediction_path": combined_model.get("prediction_path"),
                "sample_auroc": combined_model.get("sample_auroc"),
                "span_auroc": combined_model.get("span_auroc"),
            },
        },
        "baseline_errors": baseline_errors,
        "combined_errors": combined_errors,
        "baseline_error_entropy_stats": baseline_error_entropy,
        "combined_error_entropy_stats": combined_error_entropy,
        "corrections": corrections,
        "correction_entropy_stats": correction_entropy,
    }

    dump_json(output_dir / "error_analysis.json", report)

    summary_lines = [
        f"Baseline model: {baseline_key} | sample(max) AUROC={baseline_model.get('sample_auroc')}",
        f"Combined model: {combined_key} | sample(max) AUROC={combined_model.get('sample_auroc')}",
        (
            "Baseline errors: "
            f"FP={baseline_errors['counts'].get('fp', 0)} "
            f"FN={baseline_errors['counts'].get('fn', 0)}"
        ),
        (
            "Combined errors: "
            f"FP={combined_errors['counts'].get('fp', 0)} "
            f"FN={combined_errors['counts'].get('fn', 0)}"
        ),
        (
            "Correction summary: "
            f"fixed={corrections['counts'].get('corrected', 0)} "
            f"new_errors={corrections['counts'].get('introduced_errors', 0)} "
            f"remaining={corrections['counts'].get('remaining_errors', 0)}"
        ),
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "error_analysis_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text.rstrip())

    return report


__all__ = [
    "load_baseline_predictions",
    "identify_errors",
    "compute_error_entropy_stats",
    "compare_model_corrections",
    "run_error_analysis",
]
