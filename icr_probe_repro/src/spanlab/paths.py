from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = LAB_ROOT
PROJECT_ROOT = LAB_ROOT  # Backward compatibility for copied utilities.

CONFIGS_DIR = LAB_ROOT / "configs"
INPUT_DATA_DIR = LAB_ROOT / "data" / "input"

DEFAULT_ICR_OUTPUT_PATH = INPUT_DATA_DIR / "icr_halu_eval_random_qwen2.5.jsonl"
DEFAULT_HALUEVAL_QA_PATH = INPUT_DATA_DIR / "qa_data.json"

LAB_DATA_DIR = LAB_ROOT / "data"
INTERMEDIATE_DATA_DIR = LAB_DATA_DIR / "intermediate"
SPAN_CANDIDATE_DIR = LAB_DATA_DIR / "span_candidates"
SPAN_LABEL_DIR = LAB_DATA_DIR / "span_labels"
DATASET_DIR = LAB_DATA_DIR / "datasets"

RESULTS_DIR = LAB_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = LAB_ROOT / "figures"


def default_span_ready_path() -> Path:
    return INTERMEDIATE_DATA_DIR / "icr_halu_eval_span_ready.jsonl"


def default_alignment_summary_path() -> Path:
    return INTERMEDIATE_DATA_DIR / "icr_halu_eval_span_ready.summary.json"


def default_generated_icr_path() -> Path:
    return INPUT_DATA_DIR / "icr_halu_eval_recomputed.jsonl"


def default_tokenizer_window_path() -> Path:
    return SPAN_CANDIDATE_DIR / "tokenizer_windows.jsonl"


def default_spacy_span_path() -> Path:
    return SPAN_CANDIDATE_DIR / "spacy_spans.jsonl"


def default_silver_label_path(route_name: str) -> Path:
    return SPAN_LABEL_DIR / f"{route_name}_silver_labels.jsonl"


def default_dataset_path(route_name: str) -> Path:
    return DATASET_DIR / f"{route_name}_dataset.jsonl"


def default_results_summary_json_path() -> Path:
    return RESULTS_DIR / "summary.json"


def default_results_summary_csv_path() -> Path:
    return RESULTS_DIR / "summary.csv"


def default_results_summary_md_path() -> Path:
    return RESULTS_DIR / "summary.md"
