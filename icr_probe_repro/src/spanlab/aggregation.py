from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np


def aggregate_probabilities(probabilities: Sequence[float], mode: str, top_k: int = 3) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0:
        return 0.0
    if mode == "max":
        return float(probs.max())
    if mode == "topk_mean":
        k = min(top_k, probs.size)
        topk = np.partition(probs, -k)[-k:]
        return float(topk.mean())
    if mode == "noisy_or":
        return float(1.0 - np.prod(1.0 - probs))
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def aggregate_sample_predictions(
    rows: Sequence[Dict],
    probabilities: Sequence[float],
    top_k: int = 3,
) -> Dict[str, Dict[str, np.ndarray]]:
    grouped = defaultdict(list)
    labels = {}
    for row, probability in zip(rows, probabilities):
        grouped[row["sample_id"]].append(float(probability))
        labels[row["sample_id"]] = int(row["sample_label"])

    aggregated: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in ("max", "topk_mean", "noisy_or"):
        sample_ids = sorted(grouped)
        probs = np.asarray([aggregate_probabilities(grouped[sample_id], mode, top_k=top_k) for sample_id in sample_ids])
        sample_labels = np.asarray([labels[sample_id] for sample_id in sample_ids], dtype=np.int32)
        aggregated[mode] = {
            "sample_ids": np.asarray(sample_ids, dtype=object),
            "labels": sample_labels,
            "probs": probs,
        }
    return aggregated
