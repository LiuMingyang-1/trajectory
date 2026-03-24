#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.aggregation import aggregate_sample_predictions
from spanlab.evaluation import evaluate_binary_predictions
from spanlab.io_utils import dump_json, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate span-level prediction files back to sample-level metrics.")
    parser.add_argument("--prediction_files", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = {}
    for prediction_path in args.prediction_files:
        rows = read_jsonl(prediction_path)
        filtered = [row for row in rows if row.get("probability") is not None]
        probabilities = [float(row["probability"]) for row in filtered]
        aggregated = aggregate_sample_predictions(filtered, probabilities, top_k=args.top_k)
        summary[prediction_path.name] = {
            mode: evaluate_binary_predictions(payload["labels"], payload["probs"])
            for mode, payload in aggregated.items()
        }

    if args.output_path is not None:
        dump_json(args.output_path, summary)

    for name, metrics_by_mode in summary.items():
        print(f"\n{name}")
        for mode, metrics in metrics_by_mode.items():
            print(
                f"  {mode:10s} AUROC={metrics['AUROC']:.4f} "
                f"AUPRC={metrics['AUPRC']:.4f} F1={metrics['F1']:.4f}"
            )


if __name__ == "__main__":
    main()
