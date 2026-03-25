#!/usr/bin/env python3
"""Run Cut B: Shallow Overconfidence Hypothesis analysis and experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ensure_spanlab_importable


ensure_spanlab_importable()

from cut_b.analysis import analyze_shallow_deep_confidence, format_analysis_report
from cut_b.train import run_mismatch_experiment
from cut_b.visualize import (
    plot_layerwise_entropy_curve,
    plot_mismatch_distribution,
    plot_shallow_vs_deep_scatter,
)
from shared.data_loader import combined_dataset_path as default_combined_dataset_path
from shared.data_loader import load_entropy_records, load_icr_records, merge_icr_entropy
from shared.eval_utils import dump_json
from shared.paths import COMBINED_DATASET_DIR, CUT_B_RESULTS_DIR, ENTROPY_JSONL, ICR_INPUT_JSONL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=CUT_B_RESULTS_DIR, help="Directory for Cut B outputs.")
    parser.add_argument(
        "--combined_dataset",
        type=Path,
        default=None,
        help="Optional combined span dataset JSONL for training experiments.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for BaselineMLP training.")
    parser.add_argument("--skip_training", action="store_true", help="Run only analysis and visualizations.")
    return parser.parse_args()


def resolve_combined_dataset(explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path

    candidate = default_combined_dataset_path("tokenizer_windows", pooling="mean")
    return candidate if candidate.exists() else None


def ensure_required_inputs() -> None:
    if not ICR_INPUT_JSONL.exists():
        print(f"ERROR: ICR input not found at {ICR_INPUT_JSONL}")
        sys.exit(1)
    if not ENTROPY_JSONL.exists():
        print(f"ERROR: Entropy data not found at {ENTROPY_JSONL}")
        print("Run `python3 scripts/run_entropy_extraction.py` first.")
        sys.exit(1)


def main() -> None:
    args = parse_args()
    ensure_required_inputs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ICR input: {ICR_INPUT_JSONL}")
    print(f"Entropy input: {ENTROPY_JSONL}")
    print(f"Output dir: {output_dir}")

    print("\n[1/5] Loading ICR and entropy records")
    icr_records = load_icr_records()
    entropy_records = load_entropy_records()
    merged_records = merge_icr_entropy(icr_records, entropy_records)
    print(f"Merged samples: {len(merged_records)}")

    print("\n[2/5] Running Cut B statistical analysis")
    analysis_results = analyze_shallow_deep_confidence(merged_records)
    analysis_report = format_analysis_report(analysis_results)
    dump_json(output_dir / "analysis_results.json", analysis_results)
    (output_dir / "analysis_report.txt").write_text(analysis_report, encoding="utf-8")
    print(analysis_report)

    print("[3/5] Generating figures")
    figures_dir = output_dir / "figures"
    plot_layerwise_entropy_curve(merged_records, figures_dir / "layerwise_entropy_curve.png")
    plot_shallow_vs_deep_scatter(merged_records, figures_dir / "shallow_vs_deep_scatter.png")
    plot_mismatch_distribution(merged_records, figures_dir / "mismatch_distribution.png")
    print(f"Saved figures to {figures_dir}")

    training_summary = None
    combined_dataset_path = resolve_combined_dataset(args.combined_dataset)
    if args.skip_training:
        print("\n[4/5] Training skipped by request")
    elif combined_dataset_path is None or not combined_dataset_path.exists():
        print("\n[4/5] No combined span dataset found; skipping training experiments")
        print(f"Looked under {COMBINED_DATASET_DIR}")
    else:
        print("\n[4/5] Running training experiments")
        print(f"Combined dataset: {combined_dataset_path}")
        training_summary = run_mismatch_experiment(
            combined_dataset_path=combined_dataset_path,
            output_dir=output_dir / "training",
            device=args.device,
        )
        print(f"Saved training results to {output_dir / 'training'}")

    print("\n[5/5] Summary")
    shallow_test = analysis_results["tests"]["shallow_correct_vs_hallucinated"]
    deep_test = analysis_results["tests"]["deep_correct_vs_hallucinated"]
    print(
        "Key p-values: "
        f"shallow={shallow_test.get('p_value')} "
        f"deep={deep_test.get('p_value')}"
    )
    print(f"Analysis report: {output_dir / 'analysis_report.txt'}")
    print(f"Analysis JSON: {output_dir / 'analysis_results.json'}")
    print(f"Figures dir: {figures_dir}")
    if training_summary is not None:
        print(f"Training summary: {output_dir / 'training' / 'training_summary.json'}")


if __name__ == "__main__":
    main()
