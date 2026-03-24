#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-full}"
DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"
ICR_INPUT_PATH="${ICR_INPUT_PATH:-${LAB_DIR}/data/input/icr_halu_eval_random_qwen2.5.jsonl}"
QA_DATA_PATH="${QA_DATA_PATH:-${LAB_DIR}/data/input/qa_data.json}"
ICR_OUTPUT_PATH="${ICR_OUTPUT_PATH:-${LAB_DIR}/data/input/icr_halu_eval_recomputed.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
DTYPE="${DTYPE:-float16}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"

run_cmd() {
  echo "[RUN] $*"
  "$@"
}

with_optional_max_samples() {
  local -a cmd=("$@")
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  run_cmd "${cmd[@]}"
}

compute_icr() {
  if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
    echo "MODEL_NAME_OR_PATH is required for from-scratch mode." >&2
    exit 1
  fi
  with_optional_max_samples "${PYTHON_BIN}" "${LAB_DIR}/scripts/compute_icr_halueval.py" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_path "${QA_DATA_PATH}" \
    --output_path "${ICR_OUTPUT_PATH}" \
    --task qa \
    --pairing random \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}"
}

prepare_span_ready() {
  local icr_input_path="$1"
  with_optional_max_samples "${PYTHON_BIN}" "${LAB_DIR}/scripts/prepare_span_ready_data.py" \
    --input_path "${icr_input_path}" \
    --qa_data_path "${QA_DATA_PATH}"
}

build_tokenizer_route() {
  with_optional_max_samples "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_tokenizer_windows.py"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_silver_span_labels.py" \
    --span_path "${LAB_DIR}/data/span_candidates/tokenizer_windows.jsonl"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_span_dataset.py" \
    --labeled_span_path "${LAB_DIR}/data/span_labels/tokenizer_windows_silver_labels.jsonl"
}

build_spacy_route() {
  with_optional_max_samples "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_spacy_spans.py" \
    --spacy_model "${SPACY_MODEL}"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_silver_span_labels.py" \
    --span_path "${LAB_DIR}/data/span_candidates/spacy_spans.jsonl"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/build_span_dataset.py" \
    --labeled_span_path "${LAB_DIR}/data/span_labels/spacy_spans_silver_labels.jsonl"
}

train_minimal() {
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_baseline_mlp.py" \
    --dataset_path "${LAB_DIR}/data/datasets/tokenizer_windows_dataset.jsonl" \
    --device "${DEVICE}"
}

train_full_for_dataset() {
  local dataset_path="$1"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_baseline_mlp.py" \
    --dataset_path "${dataset_path}" \
    --device "${DEVICE}"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_discrepancy.py" \
    --dataset_path "${dataset_path}"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_temporal_conv.py" \
    --dataset_path "${dataset_path}" \
    --device "${DEVICE}"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_change_point.py" \
    --dataset_path "${dataset_path}"
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/train_trajectory_encoder.py" \
    --dataset_path "${dataset_path}" \
    --device "${DEVICE}"
}

summarize_results() {
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/summarize_results.py"
}

generate_figures() {
  run_cmd "${PYTHON_BIN}" "${LAB_DIR}/scripts/generate_default_figures.py"
}

show_usage() {
  cat <<EOF
Usage:
  bash icr_probe_repro/scripts/run_pipeline.sh [minimal|full|from-scratch|figures-only|summary-only]

Modes:
  minimal       Reuse existing ICR JSONL, run tokenizer-window + Baseline MLP only.
  full          Reuse existing ICR JSONL, run both span routes and all model families.
  from-scratch  Recompute ICR first, then run the full pipeline.
  figures-only  Regenerate figures from existing results.
  summary-only  Regenerate result summary tables only.

Environment variables:
  PYTHON_BIN=python3
  DEVICE=cuda|cpu
  MAX_SAMPLES=1000
  SPACY_MODEL=en_core_web_sm
  ICR_INPUT_PATH=/path/to/precomputed_icr.jsonl
  QA_DATA_PATH=/path/to/qa_data.json
  ICR_OUTPUT_PATH=/path/to/recomputed_icr.jsonl
  MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
  ATTN_IMPLEMENTATION=eager
  DTYPE=float16|bfloat16|float32

Examples:
  bash icr_probe_repro/scripts/run_pipeline.sh full
  MAX_SAMPLES=1000 DEVICE=cpu bash icr_probe_repro/scripts/run_pipeline.sh minimal
  MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct DEVICE=cuda bash icr_probe_repro/scripts/run_pipeline.sh from-scratch
EOF
}

case "${MODE}" in
  minimal)
    prepare_span_ready "${ICR_INPUT_PATH}"
    build_tokenizer_route
    train_minimal
    summarize_results
    generate_figures
    ;;
  full)
    prepare_span_ready "${ICR_INPUT_PATH}"
    build_tokenizer_route
    build_spacy_route
    train_full_for_dataset "${LAB_DIR}/data/datasets/tokenizer_windows_dataset.jsonl"
    train_full_for_dataset "${LAB_DIR}/data/datasets/spacy_spans_dataset.jsonl"
    summarize_results
    generate_figures
    ;;
  from-scratch)
    compute_icr
    prepare_span_ready "${ICR_OUTPUT_PATH}"
    build_tokenizer_route
    build_spacy_route
    train_full_for_dataset "${LAB_DIR}/data/datasets/tokenizer_windows_dataset.jsonl"
    train_full_for_dataset "${LAB_DIR}/data/datasets/spacy_spans_dataset.jsonl"
    summarize_results
    generate_figures
    ;;
  figures-only)
    generate_figures
    ;;
  summary-only)
    summarize_results
    ;;
  *)
    show_usage
    exit 1
    ;;
esac
