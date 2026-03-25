"""Load and merge ICR scores with entropy scores, and build combined span datasets."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import COMBINED_DATASET_DIR, ENTROPY_JSONL, ICR_INPUT_JSONL, ensure_spanlab_importable


ensure_spanlab_importable()

from spanlab.alignment import USABLE_LAYERS
from spanlab.io_utils import read_jsonl, write_jsonl
from spanlab.representation import pool_span_icr


RecordKey = Tuple[int, int]


def combined_dataset_path(name: str, pooling: str = "mean") -> Path:
    """Return a default output path under the combined dataset directory."""
    filename = name if name.endswith(".jsonl") else f"{name}_{pooling}.jsonl"
    return COMBINED_DATASET_DIR / filename


def load_entropy_records(path: Path = ENTROPY_JSONL) -> dict[RecordKey, dict[str, Any]]:
    """Load entropy JSONL rows and index them by ``(index, candidate_index)``."""
    records = read_jsonl(path)
    indexed: dict[RecordKey, dict[str, Any]] = {}
    for record in records:
        key = (int(record["index"]), int(record.get("candidate_index", 0)))
        if key in indexed:
            raise ValueError(
                f"Duplicate entropy record for index={key[0]} candidate_index={key[1]} in {path}."
            )
        indexed[key] = record
    return indexed


def load_icr_records(path: Path = ICR_INPUT_JSONL) -> list[dict[str, Any]]:
    """Load the base ICR JSONL rows."""
    return read_jsonl(path)


def merge_icr_entropy(
    icr_records: list[dict[str, Any]],
    entropy_records: Mapping[RecordKey, dict[str, Any]],
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Merge ICR and entropy rows by ``(index, candidate_index)``.

    By default, missing entropy features raise so downstream experiments never
    train on a silently truncated sample set.
    """
    merged: list[dict[str, Any]] = []
    missing_keys: list[RecordKey] = []

    for record in icr_records:
        key = (int(record["index"]), int(record.get("candidate_index", 0)))
        entropy_record = entropy_records.get(key)
        if entropy_record is None:
            missing_keys.append(key)
            continue

        merged.append({**record, "entropy_scores": entropy_record["entropy_scores"]})

    if missing_keys:
        preview = ", ".join(f"{index}:{candidate_index}" for index, candidate_index in missing_keys[:10])
        message = (
            f"{len(missing_keys)} ICR records had no matching entropy data. "
            "Entropy extraction must cover the same sample set before building combined features. "
            f"First missing sample ids: {preview}"
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)

    return merged


def pool_entropy_for_span(
    entropy_scores: Any,
    token_start: int,
    token_end: int,
    pooling: str = "mean",
) -> np.ndarray:
    """Pool entropy values over a token span using the spanlab pooling logic."""
    return pool_span_icr(entropy_scores, token_start, token_end, pooling=pooling)


def build_combined_span_record(
    sample_row: dict[str, Any],
    labeled_span_row: dict[str, Any],
    pooling: str = "mean",
) -> Optional[dict[str, Any]]:
    """Build a span record containing pooled ICR, entropy, and delta-entropy features."""
    token_start = int(labeled_span_row["token_start"])
    token_end = int(labeled_span_row["token_end"])

    icr_matrix = np.asarray(sample_row["icr_scores"], dtype=np.float32)
    if icr_matrix.ndim != 2:
        raise ValueError(f"Expected ICR scores with shape [layers, tokens], got {icr_matrix.shape}.")
    if icr_matrix.shape[0] < USABLE_LAYERS:
        raise ValueError(f"Expected at least {USABLE_LAYERS} ICR layers, got {icr_matrix.shape[0]}.")
    icr_matrix = icr_matrix[:USABLE_LAYERS]

    span_icr = pool_span_icr(icr_matrix, token_start, token_end, pooling=pooling)
    sample_icr = pool_span_icr(icr_matrix, 0, icr_matrix.shape[1], pooling="mean")

    if "entropy_scores" not in sample_row:
        return None

    entropy_matrix = np.asarray(sample_row["entropy_scores"], dtype=np.float32)
    ent_n_tokens = entropy_matrix.shape[1]

    if ent_n_tokens == 0:
        span_entropy = np.zeros(entropy_matrix.shape[0], dtype=np.float32)
        sample_entropy = np.zeros(entropy_matrix.shape[0], dtype=np.float32)
    else:
        ent_start = min(token_start, ent_n_tokens)
        ent_end = min(token_end, ent_n_tokens)

        if ent_end <= ent_start:
            span_entropy = np.zeros(entropy_matrix.shape[0], dtype=np.float32)
        else:
            span_entropy = pool_span_icr(entropy_matrix, ent_start, ent_end, pooling=pooling)

        sample_entropy = pool_span_icr(entropy_matrix, 0, ent_n_tokens, pooling="mean")

    span_delta_entropy = np.diff(span_entropy)

    return {
        **labeled_span_row,
        "pooling": pooling,
        "is_labeled": labeled_span_row.get("silver_label") is not None,
        "span_vector": [round(float(value), 8) for value in span_icr.tolist()],
        "sample_vector": [round(float(value), 8) for value in sample_icr.tolist()],
        "entropy_vector": [round(float(value), 8) for value in span_entropy.tolist()],
        "sample_entropy_vector": [round(float(value), 8) for value in sample_entropy.tolist()],
        "delta_entropy_vector": [round(float(value), 8) for value in span_delta_entropy.tolist()],
    }


def save_combined_span_dataset(rows: list[dict[str, Any]], dataset_name: str, pooling: str = "mean") -> Path:
    """Write combined span rows under ``data/combined`` and return the output path."""
    output_path = combined_dataset_path(dataset_name, pooling=pooling)
    write_jsonl(output_path, rows)
    return output_path


def load_combined_span_dataset(
    dataset_path: Path,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load a combined span dataset and return row metadata plus model-ready arrays.

    Returns:
        rows, icr_vectors, entropy_vectors, delta_entropy_vectors,
        silver_labels, sample_ids, sample_labels
    """
    rows = read_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"No rows in {dataset_path}")

    icr_vectors = np.asarray([row["span_vector"] for row in rows], dtype=np.float32)
    entropy_vectors = np.asarray([row["entropy_vector"] for row in rows], dtype=np.float32)
    delta_entropy_vectors = np.asarray([row["delta_entropy_vector"] for row in rows], dtype=np.float32)
    if icr_vectors.ndim != 2:
        raise ValueError(f"Expected 2D ICR vectors in {dataset_path}, got {icr_vectors.shape}.")
    if entropy_vectors.ndim != 2:
        raise ValueError(f"Expected 2D entropy vectors in {dataset_path}, got {entropy_vectors.shape}.")
    if delta_entropy_vectors.ndim != 2:
        raise ValueError(f"Expected 2D delta-entropy vectors in {dataset_path}, got {delta_entropy_vectors.shape}.")

    if icr_vectors.shape[1] != USABLE_LAYERS:
        raise ValueError(
            f"Expected combined dataset ICR vectors to have width {USABLE_LAYERS}, "
            f"got {icr_vectors.shape[1]}. Rebuild the combined dataset with `python3 scripts/run_cut_a.py`."
        )
    if entropy_vectors.shape[1] != icr_vectors.shape[1] + 1:
        raise ValueError(
            f"Expected entropy vectors width {icr_vectors.shape[1] + 1}, got {entropy_vectors.shape[1]} "
            f"in {dataset_path}."
        )
    if delta_entropy_vectors.shape[1] != icr_vectors.shape[1]:
        raise ValueError(
            f"Expected delta-entropy vectors width {icr_vectors.shape[1]}, got {delta_entropy_vectors.shape[1]} "
            f"in {dataset_path}."
        )

    silver_labels = np.asarray(
        [-1 if row.get("silver_label") is None else int(row["silver_label"]) for row in rows],
        dtype=np.int32,
    )
    sample_ids = np.asarray([row["sample_id"] for row in rows], dtype=object)
    sample_labels = np.asarray([int(row["sample_label"]) for row in rows], dtype=np.int32)

    return (
        rows,
        icr_vectors,
        entropy_vectors,
        delta_entropy_vectors,
        silver_labels,
        sample_ids,
        sample_labels,
    )


__all__ = [
    "combined_dataset_path",
    "load_entropy_records",
    "load_icr_records",
    "merge_icr_entropy",
    "pool_entropy_for_span",
    "build_combined_span_record",
    "save_combined_span_dataset",
    "load_combined_span_dataset",
]
