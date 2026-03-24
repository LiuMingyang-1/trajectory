#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.io_utils import read_jsonl
from spanlab.paths import FIGURES_DIR
from spanlab.visualization import build_prediction_index, plot_span_length_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dataset-level span statistics.")
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--prediction_file", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_rows = read_jsonl(args.dataset_path)
    prediction_by_span = None
    if args.prediction_file is not None:
        prediction_by_span = build_prediction_index(read_jsonl(args.prediction_file))

    output_path = args.output_path or (FIGURES_DIR / f"{args.dataset_path.stem}_span_stats.png")
    plot_span_length_statistics(dataset_rows, output_path=output_path, prediction_by_span=prediction_by_span)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
