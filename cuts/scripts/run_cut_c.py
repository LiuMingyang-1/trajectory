#!/usr/bin/env python3
"""Run Cut C: adaptive probe with confidence weighting."""

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

from cut_c.compare import run_gate_comparison
from cut_c.train import run_cut_c_experiment
from shared.data_loader import combined_dataset_path as default_combined_dataset_path
from shared.eval_utils import dump_json
from shared.paths import COMBINED_DATASET_DIR, CUT_C_RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=CUT_C_RESULTS_DIR, help="Directory for Cut C outputs.")
    parser.add_argument(
        "--combined_dataset",
        type=Path,
        default=None,
        help="Combined span dataset JSONL to use for Cut C training.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for Cut C training.")
    parser.add_argument("--skip_training", action="store_true", help="Reuse existing Cut C training outputs.")
    parser.add_argument("--skip_compare", action="store_true", help="Skip gate comparison analysis.")
    return parser.parse_args()


def resolve_combined_dataset(explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None

    candidate = default_combined_dataset_path("tokenizer_windows", pooling="mean")
    return candidate if candidate.exists() else None


def ensure_combined_dataset_exists(explicit_path: Path | None) -> Path:
    combined_dataset_path = resolve_combined_dataset(explicit_path)
    if combined_dataset_path is None or not combined_dataset_path.exists():
        print("ERROR: No combined span dataset found for Cut C.")
        print(f"Checked explicit path: {explicit_path}" if explicit_path is not None else f"Checked: {COMBINED_DATASET_DIR}")
        print("Build one first with `python3 scripts/run_cut_a.py` or by preparing the baseline span labels dataset.")
        sys.exit(1)
    return combined_dataset_path


def main() -> None:
    args = parse_args()
    combined_dataset_path = ensure_combined_dataset_exists(args.combined_dataset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = output_dir / "training"
    comparison_dir = output_dir / "comparison"

    print(f"Combined dataset: {combined_dataset_path}")
    print(f"Output dir: {output_dir}")

    training_summary = None
    if args.skip_training:
        print("\n[1/3] Training skipped by request")
    else:
        print("\n[1/3] Running Cut C training")
        training_summary = run_cut_c_experiment(
            combined_dataset_path=combined_dataset_path,
            output_dir=training_dir,
            device=args.device,
        )
        print(f"Saved training results to {training_dir}")

    comparison_summary = None
    if args.skip_compare:
        print("\n[2/3] Gate comparison skipped by request")
    else:
        try:
            print("\n[2/3] Running gate comparison analysis")
            comparison_summary = run_gate_comparison(
                combined_dataset_path=combined_dataset_path,
                training_dir=training_dir,
                output_dir=comparison_dir,
            )
            print(f"Saved gate analysis to {comparison_dir}")
        except FileNotFoundError as exc:
            print(f"\n[2/3] Gate comparison skipped: {exc}")

    summary = {
        "combined_dataset_path": str(combined_dataset_path),
        "training_dir": str(training_dir) if training_dir.exists() else None,
        "training_summary_path": str(training_dir / "training_summary.json")
        if (training_dir / "training_summary.json").exists()
        else None,
        "comparison_dir": str(comparison_dir) if comparison_dir.exists() else None,
        "comparison_summary_path": str(comparison_dir / "comparison_summary.json")
        if (comparison_dir / "comparison_summary.json").exists()
        else None,
        "best_model": training_summary.get("best_model") if training_summary is not None else None,
        "primary_gate_model": comparison_summary.get("primary_model") if comparison_summary is not None else None,
    }
    dump_json(output_dir / "run_summary.json", summary)

    print("\n[3/3] Summary")
    if training_summary is not None:
        best_model = training_summary["best_model"]
        print(
            "Best Cut C model: "
            f"{best_model['feature_set']} / {best_model['model']} "
            f"(sample AUROC={best_model['sample_auroc']})"
        )
        print(f"Training summary: {training_dir / 'training_summary.json'}")
    if comparison_summary is not None:
        primary = comparison_summary["primary_model"]
        print(
            "Primary gate analysis: "
            f"{primary['feature_set']} / {primary['model']} "
            f"(sample AUROC={primary['overall_sample_auroc']})"
        )
        print(f"Comparison summary: {comparison_dir / 'comparison_summary.json'}")
    print(f"Run summary: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
