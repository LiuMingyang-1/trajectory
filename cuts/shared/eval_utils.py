"""Evaluation helpers re-exported from ``spanlab``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ensure_spanlab_importable


ensure_spanlab_importable()

from spanlab.aggregation import aggregate_probabilities, aggregate_sample_predictions
from spanlab.evaluation import (
    average_precision_binary,
    build_group_folds,
    evaluate_binary_predictions,
    roc_auc_binary,
    summarize_metric_dicts,
)
from spanlab.io_utils import dump_json, ensure_parent_dir, read_jsonl, write_jsonl


def print_metrics_summary(metrics: dict[str, Any], prefix: str = "") -> None:
    """Pretty-print a metric summary dictionary."""
    if prefix:
        print(f"\n{'=' * 60}")
        print(f"  {prefix}")
        print(f"{'=' * 60}")

    if "span_level" in metrics:
        span_metrics = metrics["span_level"]
        print(
            "  Span-level:  "
            f"AUROC={span_metrics.get('AUROC_mean', 0):.4f}±{span_metrics.get('AUROC_std', 0):.4f}  "
            f"AUPRC={span_metrics.get('AUPRC_mean', 0):.4f}  "
            f"F1={span_metrics.get('F1_mean', 0):.4f}"
        )

    if "sample_level" in metrics:
        for mode in ("max", "topk_mean", "noisy_or"):
            if mode not in metrics["sample_level"]:
                continue
            sample_metrics = metrics["sample_level"][mode]
            print(
                f"  Sample({mode}): "
                f"AUROC={sample_metrics.get('AUROC_mean', 0):.4f}±{sample_metrics.get('AUROC_std', 0):.4f}  "
                f"AUPRC={sample_metrics.get('AUPRC_mean', 0):.4f}  "
                f"F1={sample_metrics.get('F1_mean', 0):.4f}"
            )


__all__ = [
    "evaluate_binary_predictions",
    "summarize_metric_dicts",
    "build_group_folds",
    "roc_auc_binary",
    "average_precision_binary",
    "aggregate_sample_predictions",
    "aggregate_probabilities",
    "read_jsonl",
    "write_jsonl",
    "dump_json",
    "ensure_parent_dir",
    "print_metrics_summary",
]
