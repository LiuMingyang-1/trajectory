#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.io_utils import read_jsonl
from spanlab.paths import FIGURES_DIR, default_span_ready_path
from spanlab.visualization import plot_case_heatmap, select_case_sample_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a single sample heatmap from span predictions.")
    parser.add_argument("--span_ready_path", type=Path, default=default_span_ready_path())
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--prediction_file", type=Path, required=True)
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--source_sample_index", type=int, default=None)
    parser.add_argument(
        "--selection",
        type=str,
        default="highest_hallucinated",
        choices=["highest_hallucinated", "highest_false_positive", "lowest_hallucinated"],
    )
    parser.add_argument("--aggregation_mode", type=str, default="noisy_or", choices=["max", "topk_mean", "noisy_or"])
    parser.add_argument("--output_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    span_ready_rows = read_jsonl(args.span_ready_path)
    dataset_rows = read_jsonl(args.dataset_path)
    prediction_rows = read_jsonl(args.prediction_file)

    sample_lookup = {row["sample_id"]: row for row in span_ready_rows}
    if args.sample_id is not None:
        sample_id = args.sample_id
    elif args.source_sample_index is not None:
        match = next(row for row in span_ready_rows if int(row["source_sample_index"]) == args.source_sample_index)
        sample_id = match["sample_id"]
    else:
        sample_id = select_case_sample_id(
            prediction_rows,
            selection=args.selection,
            aggregation_mode=args.aggregation_mode,
        )

    sample_row = sample_lookup[sample_id]
    sample_dataset_rows = [row for row in dataset_rows if row["sample_id"] == sample_id]
    sample_prediction_rows = [row for row in prediction_rows if row["sample_id"] == sample_id]

    output_path = args.output_path or (FIGURES_DIR / f"{sample_id.replace(':', '_')}_{args.selection}.png")
    plot_case_heatmap(
        sample_row=sample_row,
        dataset_rows=sample_dataset_rows,
        prediction_rows=sample_prediction_rows,
        output_path=output_path,
        aggregation_mode=args.aggregation_mode,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
