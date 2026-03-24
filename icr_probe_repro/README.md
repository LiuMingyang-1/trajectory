# ICR Probe Repro

This directory is a self-contained extraction of the reusable parts of the current repository for running ICR Probe experiments on HaluEval QA.

It keeps three things together in one place:

1. ICR score computation based on the original `icr_probe` implementation.
2. Span-level / tokenizer-window dataset construction from an ICR JSONL file.
3. Training, evaluation, aggregation, plotting, and result summarization for the current method families.

## Included Inputs

The directory already ships with the two inputs needed for the reuse-first workflow:

- `data/input/icr_halu_eval_random_qwen2.5.jsonl`
- `data/input/qa_data.json`

So after copying this directory elsewhere, you can usually start with:

```bash
bash scripts/run_pipeline.sh full
```

That path reuses the existing ICR output and reruns the downstream experiments only.
The default execution path is GPU-first. Override with `DEVICE=cpu` only when you explicitly want a CPU smoke test.

## What Gets Run

Two span routes:

- Tokenizer Window
- spaCy Span

Five method families:

- Baseline MLP
- Discrepancy
- Change Point
- Temporal Conv
- Trajectory Encoder

## Outputs

- `data/intermediate/`: token alignment / span-ready records
- `data/span_candidates/`: tokenizer-window and spaCy span candidates
- `data/span_labels/`: heuristic silver labels
- `data/datasets/`: trainable span datasets
- `results/`: metrics, OOF predictions, summary tables
- `figures/`: comparison figures and case heatmaps

## Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
bash scripts/run_pipeline.sh full
```

For detailed commands, see `RUN.md`.
