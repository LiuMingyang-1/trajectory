from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def roc_auc_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true_arr.sum())
    n_neg = int(len(y_true_arr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = _average_ranks(y_score_arr)
    rank_sum_pos = ranks[y_true_arr == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    total_pos = int(y_true_arr.sum())
    if total_pos == 0:
        return 0.0

    order = np.argsort(-y_score_arr, kind="mergesort")
    y_sorted = y_true_arr[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float((precision[y_sorted == 1]).sum() / total_pos)


def evaluate_binary_predictions(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    thresholds = np.unique(np.concatenate([y_prob_arr, np.array([0.5])]))
    best_f1 = 0.0
    best_threshold = 0.5
    best_accuracy = 0.0
    for threshold in thresholds:
        preds = (y_prob_arr >= threshold).astype(np.int32)
        tp = int(((preds == 1) & (y_true_arr == 1)).sum())
        fp = int(((preds == 1) & (y_true_arr == 0)).sum())
        fn = int(((preds == 0) & (y_true_arr == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        accuracy = float((preds == y_true_arr).mean())
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
            best_accuracy = accuracy

    return {
        "AUROC": roc_auc_binary(y_true_arr, y_prob_arr),
        "AUPRC": average_precision_binary(y_true_arr, y_prob_arr),
        "F1": best_f1,
        "Accuracy": best_accuracy,
        "Threshold": best_threshold,
    }


def summarize_metric_dicts(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_rows:
        return {}

    summary: Dict[str, float] = {}
    for key in metric_rows[0]:
        if key == "Threshold":
            continue
        values = np.asarray([row[key] for row in metric_rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


def build_group_folds(
    sample_ids: Sequence[str],
    sample_labels: Sequence[int],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[set, set]]:
    rng = np.random.default_rng(seed)
    grouped_counts = Counter(sample_ids)
    label_by_sample: Dict[str, int] = {}
    for sample_id, label in zip(sample_ids, sample_labels):
        label_by_sample.setdefault(sample_id, int(label))

    samples_by_label = defaultdict(list)
    for sample_id, label in label_by_sample.items():
        samples_by_label[label].append(sample_id)

    folds = [{"samples": set(), "label_counts": Counter(), "row_count": 0} for _ in range(n_splits)]

    for label, grouped_samples in samples_by_label.items():
        shuffled = list(grouped_samples)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda sample_id: grouped_counts[sample_id], reverse=True)

        for sample_id in shuffled:
            target_fold = min(
                range(n_splits),
                key=lambda fold_idx: (
                    folds[fold_idx]["label_counts"][label],
                    folds[fold_idx]["row_count"],
                ),
            )
            folds[target_fold]["samples"].add(sample_id)
            folds[target_fold]["label_counts"][label] += 1
            folds[target_fold]["row_count"] += grouped_counts[sample_id]

    all_samples = set(label_by_sample.keys())
    return [
        (all_samples - fold_state["samples"], set(fold_state["samples"]))
        for fold_state in folds
    ]
