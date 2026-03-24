# Run Guide

## 1. Install Dependencies

```bash
cd /path/to/icr_probe_repro
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

## 2. Recommended Modes

### Reuse existing ICR output

This is the default and fastest path.

```bash
bash scripts/run_pipeline.sh full
```

### Small smoke test

```bash
MAX_SAMPLES=1000 DEVICE=cpu bash scripts/run_pipeline.sh minimal
```

### Recompute ICR from scratch

This requires a model path and the full runtime dependencies for `torch` and `transformers`.

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct DEVICE=cuda \
bash scripts/run_pipeline.sh from-scratch
```

## 3. Main Inputs

- `data/input/icr_halu_eval_random_qwen2.5.jsonl`
- `data/input/qa_data.json`

You can override them with:

- `ICR_INPUT_PATH=/path/to/icr.jsonl`
- `QA_DATA_PATH=/path/to/qa_data.json`

## 4. Main Outputs

- `results/summary.json`
- `results/summary.csv`
- `results/summary.md`
- `figures/method_summary.png`
- `figures/sample_aggregation_summary.png`

## 5. Useful Commands

Only regenerate summary tables:

```bash
bash scripts/run_pipeline.sh summary-only
```

Only regenerate figures:

```bash
bash scripts/run_pipeline.sh figures-only
```

Run a single training family manually:

```bash
python3 scripts/train_baseline_mlp.py \
  --dataset_path data/datasets/tokenizer_windows_dataset.jsonl \
  --device cuda
```
