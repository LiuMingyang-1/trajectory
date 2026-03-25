# trajectory

Cloud-first experiment repo for the trajectory / ICR probe workflow.

## One-Command Start

After cloning on a GPU server:

```bash
bash run_cloud.sh
```

That command will:

- create a local `.venv` if needed
- install `icr_probe_repro/requirements.txt`
- run the full `cuts` pipeline end to end

Useful environment variables:

```bash
DEVICE=cuda
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
MAX_SAMPLES=128
PYTHON_BIN=python3
VENV_DIR=/path/to/.venv
```

Example:

```bash
DEVICE=cuda MODEL_NAME_OR_PATH=/path/to/local/model bash run_cloud.sh
```

Generated environments, intermediate data, and result files are intentionally excluded from git.
