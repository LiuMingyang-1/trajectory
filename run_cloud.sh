#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cmd=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/cuts/scripts/run_all.py"
  "--device" "${DEVICE}"
  "--python" "${PYTHON_BIN}"
  "--model_name_or_path" "${MODEL_NAME_OR_PATH}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=("--max_samples" "${MAX_SAMPLES}")
fi

cmd+=("$@")

printf "[RUN]"
printf " %q" "${cmd[@]}"
printf "\n"
"${cmd[@]}"
