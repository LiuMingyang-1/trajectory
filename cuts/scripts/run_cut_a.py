#!/usr/bin/env python3
"""Run Cut A: confidence trajectory feature experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ensure_spanlab_importable


ensure_spanlab_importable()

from cut_a.error_analysis import run_error_analysis
from cut_a.train import run_cut_a_experiment
from shared.data_loader import (
    build_combined_span_record,
    combined_dataset_path as default_combined_dataset_path,
    load_combined_span_dataset,
    load_entropy_records,
    load_icr_records,
    merge_icr_entropy,
)
from shared.eval_utils import dump_json, read_jsonl, write_jsonl
from shared.paths import (
    BASELINE_RESULTS_DIR,
    COMBINED_DATASET_DIR,
    CUT_A_RESULTS_DIR,
    ENTROPY_JSONL,
    ICR_INPUT_JSONL,
    ICR_RESULTS_DIR,
    ICR_SPAN_LABEL_DIR,
)
from spanlab.alignment import build_sample_id


DEFAULT_SPAN_LABEL_PATH = ICR_SPAN_LABEL_DIR / "tokenizer_windows_silver_labels.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=CUT_A_RESULTS_DIR, help="Directory for Cut A outputs.")
    parser.add_argument(
        "--combined_dataset",
        type=Path,
        default=None,
        help="Optional combined span dataset JSONL. If missing, the script will reuse or build one.",
    )
    parser.add_argument(
        "--span_labels",
        type=Path,
        default=DEFAULT_SPAN_LABEL_PATH,
        help="Silver-labeled span JSONL used to build the combined dataset when needed.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tokenizer_windows",
        help="Dataset stem used when writing a newly built combined span dataset.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "topk_mean"],
        help="Pooling strategy for span-level combined features.",
    )
    parser.add_argument(
        "--baseline_dir",
        type=Path,
        default=None,
        help="Optional baseline predictions directory for error analysis.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for BaselineMLP training.")
    parser.add_argument("--skip_training", action="store_true", help="Skip Cut A model training.")
    parser.add_argument("--skip_error_analysis", action="store_true", help="Skip post-training error analysis.")
    return parser.parse_args()


def ensure_required_inputs() -> None:
    if not ICR_INPUT_JSONL.exists():
        print(f"ERROR: ICR input not found at {ICR_INPUT_JSONL}")
        sys.exit(1)
    if not ENTROPY_JSONL.exists():
        print(f"ERROR: Entropy data not found at {ENTROPY_JSONL}")
        print("Run `python3 scripts/run_entropy_extraction.py` first.")
        sys.exit(1)


def _has_prediction_files(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    if path.is_file():
        return path.name.endswith(".oof_predictions.jsonl")
    return any(path.rglob("*.oof_predictions.jsonl"))


def resolve_combined_dataset(explicit_path: Path | None, dataset_name: str, pooling: str) -> Path | None:
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None

    candidate = default_combined_dataset_path(dataset_name, pooling=pooling)
    return candidate if candidate.exists() else None


def _resolved_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _load_combined_dataset_summary(path: Path) -> dict[str, object] | None:
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def combined_dataset_request_issues(
    path: Path,
    *,
    pooling: str,
    span_labels_path: Path,
) -> list[str]:
    issues: list[str] = []
    summary = _load_combined_dataset_summary(path)
    if summary is not None:
        if summary.get("pooling") != pooling:
            issues.append(
                f"summary pooling is {summary.get('pooling')!r}, expected {pooling!r}"
            )
        summary_span_labels = summary.get("span_labels_path")
        if summary_span_labels is not None and _resolved_path(summary_span_labels) != _resolved_path(span_labels_path):
            issues.append(
                "summary span_labels_path does not match the requested span labels file"
            )
        summary_icr_path = summary.get("icr_path")
        if summary_icr_path is not None and _resolved_path(summary_icr_path) != _resolved_path(ICR_INPUT_JSONL):
            issues.append("summary icr_path does not match the current ICR input file")
        summary_entropy_path = summary.get("entropy_path")
        if summary_entropy_path is not None and _resolved_path(summary_entropy_path) != _resolved_path(ENTROPY_JSONL):
            issues.append("summary entropy_path does not match the current entropy file")
        return issues

    rows = read_jsonl(path)
    if not rows:
        issues.append("dataset file is empty")
        return issues

    row_pooling = rows[0].get("pooling")
    if row_pooling != pooling:
        issues.append(f"dataset rows were built with pooling={row_pooling!r}, expected {pooling!r}")
    return issues


def build_combined_dataset(
    output_path: Path | None,
    span_labels_path: Path,
    dataset_name: str,
    pooling: str,
) -> tuple[Path, dict[str, object]]:
    if not span_labels_path.exists():
        raise FileNotFoundError(
            f"Span labels not found at {span_labels_path}. Run `python3 scripts/run_baseline.py` first."
        )

    icr_records = load_icr_records(ICR_INPUT_JSONL)
    entropy_records = load_entropy_records(ENTROPY_JSONL)
    merged_records = merge_icr_entropy(icr_records, entropy_records)
    merged_lookup = {
        build_sample_id(int(record["index"]), int(record.get("candidate_index", 0))): record for record in merged_records
    }
    labeled_rows = read_jsonl(span_labels_path)

    combined_rows: list[dict[str, object]] = []
    missing_sample_ids: list[str] = []
    for labeled_row in labeled_rows:
        sample_id = str(labeled_row["sample_id"])
        sample_row = merged_lookup.get(sample_id)
        if sample_row is None:
            missing_sample_ids.append(sample_id)
            continue
        combined_record = build_combined_span_record(sample_row, labeled_row, pooling=pooling)
        if combined_record is not None:
            combined_rows.append(combined_record)

    if not combined_rows:
        raise ValueError("Combined dataset build produced zero rows.")

    if output_path is None:
        output_path = default_combined_dataset_path(dataset_name, pooling=pooling)

    summary = {
        "output_path": str(output_path),
        "pooling": pooling,
        "dataset_name": dataset_name,
        "span_labels_path": str(span_labels_path),
        "icr_path": str(ICR_INPUT_JSONL),
        "entropy_path": str(ENTROPY_JSONL),
        "n_icr_records": len(icr_records),
        "n_entropy_records": len(entropy_records),
        "n_merged_records": len(merged_records),
        "n_span_rows": len(labeled_rows),
        "n_combined_rows": len(combined_rows),
        "n_missing_sample_ids": len(missing_sample_ids),
        "missing_sample_ids": missing_sample_ids[:100],
    }
    if missing_sample_ids:
        summary["status"] = "failed"
        summary["error"] = (
            "Some span labels reference sample_ids that are missing from the strict ICR+entropy merge. "
            "Rebuild the entropy file or baseline span labels from the same sample set."
        )
        dump_json(output_path.with_suffix(".summary.json"), summary)
        raise ValueError(
            f"{len(missing_sample_ids)} span rows reference sample_ids that are missing from the strict "
            f"ICR+entropy merge. First missing sample ids: {', '.join(missing_sample_ids[:10])}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, combined_rows)
    summary["status"] = "ok"
    dump_json(output_path.with_suffix(".summary.json"), summary)
    return output_path, summary


def resolve_baseline_dir(explicit_path: Path | None) -> Path | None:
    candidates = [explicit_path] if explicit_path is not None else [BASELINE_RESULTS_DIR, ICR_RESULTS_DIR]
    for candidate in candidates:
        if candidate is not None and _has_prediction_files(candidate):
            return candidate
    return None


def combined_dataset_is_compatible(path: Path) -> bool:
    try:
        load_combined_span_dataset(path)
    except Exception as exc:
        print(f"Existing combined dataset is incompatible and will be rebuilt: {path}")
        print(f"Reason: {exc}")
        return False
    return True


def main() -> None:
    args = parse_args()
    ensure_required_inputs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = output_dir / "training"
    error_analysis_dir = output_dir / "error_analysis"

    print(f"ICR input: {ICR_INPUT_JSONL}")
    print(f"Entropy input: {ENTROPY_JSONL}")
    print(f"Output dir: {output_dir}")

    print("\n[1/4] Resolving combined span dataset")
    requested_dataset_path = (
        Path(args.combined_dataset)
        if args.combined_dataset is not None
        else default_combined_dataset_path(args.dataset_name, pooling=args.pooling)
    )
    reusable_dataset_path = resolve_combined_dataset(args.combined_dataset, args.dataset_name, args.pooling)
    combined_dataset_path = reusable_dataset_path
    combined_dataset_summary = None
    request_issues: list[str] = []
    if combined_dataset_path is not None and combined_dataset_path.exists():
        request_issues = combined_dataset_request_issues(
            combined_dataset_path,
            pooling=args.pooling,
            span_labels_path=args.span_labels,
        )
    if (
        combined_dataset_path is not None
        and combined_dataset_path.exists()
        and combined_dataset_is_compatible(combined_dataset_path)
        and not request_issues
    ):
        print(f"Using existing combined dataset: {combined_dataset_path}")
    else:
        if request_issues:
            print(f"Existing combined dataset will be rebuilt: {combined_dataset_path}")
            for issue in request_issues:
                print(f"Reason: {issue}")
        combined_dataset_path, combined_dataset_summary = build_combined_dataset(
            output_path=requested_dataset_path,
            span_labels_path=args.span_labels,
            dataset_name=args.dataset_name,
            pooling=args.pooling,
        )
        print(f"Built combined dataset: {combined_dataset_path}")
        print(f"Combined rows: {combined_dataset_summary['n_combined_rows']}")

    training_summary = None
    if args.skip_training:
        print("\n[2/4] Training skipped by request")
    else:
        print("\n[2/4] Running Cut A training experiments")
        print(f"Combined dataset: {combined_dataset_path}")
        training_summary = run_cut_a_experiment(
            combined_dataset_path=combined_dataset_path,
            output_dir=training_dir,
            device=args.device,
        )
        print(f"Saved training results to {training_dir}")

    error_analysis_summary = None
    baseline_dir = resolve_baseline_dir(args.baseline_dir)
    if args.skip_error_analysis:
        print("\n[3/4] Error analysis skipped by request")
    elif baseline_dir is None:
        print("\n[3/4] No baseline prediction files found; skipping error analysis")
        print(f"Checked: {args.baseline_dir or BASELINE_RESULTS_DIR} and {ICR_RESULTS_DIR}")
    elif not _has_prediction_files(training_dir):
        print("\n[3/4] No Cut A prediction files found; skipping error analysis")
        print(f"Expected training outputs under {training_dir}")
    else:
        print("\n[3/4] Running baseline-vs-combined error analysis")
        print(f"Baseline predictions: {baseline_dir}")
        print(f"Combined predictions: {training_dir}")
        error_analysis_summary = run_error_analysis(
            baseline_dir=baseline_dir,
            combined_dir=training_dir,
            entropy_path=ENTROPY_JSONL,
            icr_path=ICR_INPUT_JSONL,
            output_dir=error_analysis_dir,
        )
        print(f"Saved error analysis to {error_analysis_dir}")

    summary = {
        "combined_dataset_path": str(combined_dataset_path),
        "combined_dataset_summary": combined_dataset_summary,
        "training_dir": str(training_dir),
        "training_summary_path": str(training_dir / "training_summary.json") if training_dir.exists() else None,
        "baseline_dir": str(baseline_dir) if baseline_dir is not None else None,
        "error_analysis_dir": str(error_analysis_dir) if error_analysis_dir.exists() else None,
        "best_model": training_summary.get("best_model") if training_summary is not None else None,
        "selected_models": error_analysis_summary.get("selected_models") if error_analysis_summary is not None else None,
    }
    dump_json(output_dir / "run_summary.json", summary)

    print("\n[4/4] Summary")
    print(f"Combined dataset: {combined_dataset_path}")
    if training_summary is not None:
        best_model = training_summary["best_model"]
        print(
            "Best Cut A model: "
            f"{best_model['feature_set']} / {best_model['family_group']} / {best_model['model']} "
            f"(sample AUROC={best_model['sample_auroc']})"
        )
        print(f"Training summary: {training_dir / 'training_summary.json'}")
    if error_analysis_summary is not None:
        selected = error_analysis_summary["selected_models"]
        print(
            "Error analysis models: "
            f"baseline={selected['baseline']['key']} "
            f"combined={selected['combined']['key']}"
        )
        print(f"Error analysis JSON: {error_analysis_dir / 'error_analysis.json'}")
    print(f"Run summary: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
