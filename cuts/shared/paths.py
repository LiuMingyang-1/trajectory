"""Filesystem paths shared across the cuts experiment project."""

from __future__ import annotations

from pathlib import Path


CUTS_ROOT = Path(__file__).resolve().parents[1]
ICR_REPO_ROOT = CUTS_ROOT.parent / "icr_probe_repro"
ICR_SRC_ROOT = ICR_REPO_ROOT / "src"

# Input data from ICR repo
ICR_INPUT_JSONL = ICR_REPO_ROOT / "data" / "input" / "icr_halu_eval_random_qwen2.5.jsonl"
QA_DATA_JSON = ICR_REPO_ROOT / "data" / "input" / "qa_data.json"

# ICR repo intermediate data (used by baseline)
ICR_INTERMEDIATE_DIR = ICR_REPO_ROOT / "data" / "intermediate"
ICR_SPAN_CANDIDATE_DIR = ICR_REPO_ROOT / "data" / "span_candidates"
ICR_SPAN_LABEL_DIR = ICR_REPO_ROOT / "data" / "span_labels"
ICR_DATASET_DIR = ICR_REPO_ROOT / "data" / "datasets"
ICR_RESULTS_DIR = ICR_REPO_ROOT / "results"

# Cuts project data
DATA_DIR = CUTS_ROOT / "data"
ENTROPY_DIR = DATA_DIR / "entropy"
ENTROPY_JSONL = ENTROPY_DIR / "entropy_scores.jsonl"
RESULTS_DIR = DATA_DIR / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
CUT_A_RESULTS_DIR = RESULTS_DIR / "cut_a"
CUT_B_RESULTS_DIR = RESULTS_DIR / "cut_b"
CUT_C_RESULTS_DIR = RESULTS_DIR / "cut_c"

# Combined dataset
COMBINED_DATASET_DIR = DATA_DIR / "combined"


def ensure_spanlab_importable() -> None:
    """Add the ICR repo's source directory to ``sys.path`` if needed."""
    import sys

    icr_src_root = str(ICR_SRC_ROOT)
    if icr_src_root not in sys.path:
        sys.path.insert(0, icr_src_root)


__all__ = [
    "CUTS_ROOT",
    "ICR_REPO_ROOT",
    "ICR_SRC_ROOT",
    "ICR_INPUT_JSONL",
    "QA_DATA_JSON",
    "ICR_INTERMEDIATE_DIR",
    "ICR_SPAN_CANDIDATE_DIR",
    "ICR_SPAN_LABEL_DIR",
    "ICR_DATASET_DIR",
    "ICR_RESULTS_DIR",
    "DATA_DIR",
    "ENTROPY_DIR",
    "ENTROPY_JSONL",
    "RESULTS_DIR",
    "BASELINE_RESULTS_DIR",
    "CUT_A_RESULTS_DIR",
    "CUT_B_RESULTS_DIR",
    "CUT_C_RESULTS_DIR",
    "COMBINED_DATASET_DIR",
    "ensure_spanlab_importable",
]
