#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.paths import FIGURES_DIR, RESULTS_DIR
from spanlab.visualization import load_metric_records, plot_aggregation_summary, plot_method_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot method-level metric comparisons from results/*.metrics.json.")
    parser.add_argument("--results_root", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures_dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_metric_records(args.results_root)
    if not records:
        raise ValueError(f"No metric files found under {args.results_root}")

    plot_method_summary(records, args.figures_dir / "method_summary.png")
    plot_aggregation_summary(records, args.figures_dir / "sample_aggregation_summary.png")
    print(f"Saved: {args.figures_dir / 'method_summary.png'}")
    print(f"Saved: {args.figures_dir / 'sample_aggregation_summary.png'}")


if __name__ == "__main__":
    main()
