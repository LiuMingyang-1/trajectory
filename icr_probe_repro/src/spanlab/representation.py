from typing import Any, Dict

import numpy as np


def pool_span_icr(icr_scores: Any, token_start: int, token_end: int, pooling: str = "mean") -> np.ndarray:
    matrix = np.asarray(icr_scores, dtype=np.float32)
    window = matrix[:, token_start:token_end]
    if window.ndim != 2 or window.shape[1] == 0:
        raise ValueError(f"Invalid span window [{token_start}, {token_end}) for ICR matrix with shape {matrix.shape}.")

    if pooling == "mean":
        return window.mean(axis=1)
    if pooling == "max":
        return window.max(axis=1)
    if pooling == "topk_mean":
        k = min(2, window.shape[1])
        topk = np.partition(window, -k, axis=1)[:, -k:]
        return topk.mean(axis=1)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def build_span_dataset_record(sample_row: Dict[str, Any], labeled_span_row: Dict[str, Any], pooling: str = "mean") -> Dict[str, Any]:
    token_start = int(labeled_span_row["token_start"])
    token_end = int(labeled_span_row["token_end"])
    span_vector = pool_span_icr(sample_row["icr_scores"], token_start, token_end, pooling=pooling)
    sample_vector = pool_span_icr(sample_row["icr_scores"], 0, len(sample_row["response_token_ids"]), pooling="mean")

    return {
        **labeled_span_row,
        "pooling": pooling,
        "is_labeled": labeled_span_row.get("silver_label") is not None,
        "span_vector": [round(float(value), 8) for value in span_vector.tolist()],
        "sample_vector": [round(float(value), 8) for value in sample_vector.tolist()],
    }
