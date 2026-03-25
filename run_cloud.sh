#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
DEVICE="${DEVICE:-cuda}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[SETUP] Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"

echo "[SETUP] Installing Python dependencies"
"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install -r "${ROOT_DIR}/icr_probe_repro/requirements.txt"

cmd=(
  "${VENV_PYTHON}"
  "${ROOT_DIR}/cuts/scripts/run_all.py"
  "--device" "${DEVICE}"
  "--python" "${VENV_PYTHON}"
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
