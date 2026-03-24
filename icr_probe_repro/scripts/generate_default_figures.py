#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.io_utils import read_jsonl
from spanlab.paths import DATASET_DIR, FIGURES_DIR, RESULTS_DIR, default_span_ready_path
from spanlab.visualization import (
    build_prediction_index,
    load_metric_records,
    plot_aggregation_summary,
    plot_case_heatmap,
    plot_method_summary,
    plot_span_length_statistics,
    select_case_sample_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a default set of span-lab visualizations.")
    parser.add_argument("--results_root", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--datasets_root", type=Path, default=DATASET_DIR)
    parser.add_argument("--span_ready_path", type=Path, default=default_span_ready_path())
    parser.add_argument("--prediction_file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    metric_records = load_metric_records(args.results_root)
    if metric_records:
        plot_method_summary(metric_records, args.figures_dir / "method_summary.png")
        plot_aggregation_summary(metric_records, args.figures_dir / "sample_aggregation_summary.png")
        print(f"Saved: {args.figures_dir / 'method_summary.png'}")
        print(f"Saved: {args.figures_dir / 'sample_aggregation_summary.png'}")

    prediction_files = [args.prediction_file] if args.prediction_file is not None else sorted(
        args.results_root.rglob("*.oof_predictions.jsonl")
    )
    if not prediction_files:
        return

    span_ready_rows = read_jsonl(args.span_ready_path) if args.span_ready_path.exists() else []
    sample_lookup = {row["sample_id"]: row for row in span_ready_rows}
    dataset_cache = {}

    for prediction_file in prediction_files:
        if prediction_file is None or not prediction_file.exists():
            continue

        dataset_name = prediction_file.parent.parent.name
        model_name = prediction_file.name.replace(".oof_predictions.jsonl", "")
        dataset_path = args.datasets_root / f"{dataset_name}.jsonl"
        if not dataset_path.exists():
            print(f"Skip {prediction_file}: dataset not found -> {dataset_path}")
            continue

        if dataset_path not in dataset_cache:
            dataset_cache[dataset_path] = read_jsonl(dataset_path)
        dataset_rows = dataset_cache[dataset_path]
        prediction_rows = read_jsonl(prediction_file)
        prediction_by_span = build_prediction_index(prediction_rows)

        prefix = f"{dataset_name}__{model_name}"
        span_stats_path = args.figures_dir / f"{prefix}_span_stats.png"
        plot_span_length_statistics(
            dataset_rows,
            output_path=span_stats_path,
            prediction_by_span=prediction_by_span,
        )
        print(f"Saved: {span_stats_path}")

        if sample_lookup:
            for selection in ["highest_hallucinated", "highest_false_positive"]:
                sample_id = select_case_sample_id(prediction_rows, selection=selection, aggregation_mode="noisy_or")
                case_path = args.figures_dir / f"{prefix}_{selection}.png"
                plot_case_heatmap(
                    sample_row=sample_lookup[sample_id],
                    dataset_rows=[row for row in dataset_rows if row["sample_id"] == sample_id],
                    prediction_rows=[row for row in prediction_rows if row["sample_id"] == sample_id],
                    output_path=case_path,
                )
                print(f"Saved: {case_path}")


if __name__ == "__main__":
    main()
