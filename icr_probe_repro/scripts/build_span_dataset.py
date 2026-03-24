#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.io_utils import dump_json, read_jsonl, write_jsonl
from spanlab.paths import default_dataset_path, default_span_ready_path
from spanlab.representation import build_span_dataset_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pool span-level ICR vectors and export a trainable dataset.")
    parser.add_argument("--span_ready_path", type=Path, default=default_span_ready_path())
    parser.add_argument("--labeled_span_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--summary_path", type=Path, default=None)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "topk_mean"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_rows = {row["sample_id"]: row for row in read_jsonl(args.span_ready_path)}
    labeled_rows = read_jsonl(args.labeled_span_path)

    route_name = args.labeled_span_path.stem.replace("_silver_labels", "")
    output_path = args.output_path or default_dataset_path(route_name)
    summary_path = args.summary_path or output_path.with_suffix(".summary.json")

    dataset_rows = []
    stats = Counter()
    for labeled_row in labeled_rows:
        sample_row = sample_rows[labeled_row["sample_id"]]
        dataset_row = build_span_dataset_record(sample_row, labeled_row, pooling=args.pooling)
        dataset_rows.append(dataset_row)
        stats["total_rows"] += 1
        stats["labeled_rows"] += int(dataset_row["is_labeled"])
        stats[f"route_{dataset_row['route']}"] += 1
        stats[f"span_type_{dataset_row['span_type']}"] += 1
        if dataset_row["silver_label"] is not None:
            stats[f"label_{dataset_row['silver_label']}"] += 1

    write_jsonl(output_path, dataset_rows)
    dump_json(
        summary_path,
        {
            "span_ready_path": str(args.span_ready_path),
            "labeled_span_path": str(args.labeled_span_path),
            "output_path": str(output_path),
            "pooling": args.pooling,
            **stats,
        },
    )
    print(f"Saved span dataset to: {output_path}")
    print(f"Rows: {stats['total_rows']} | labeled rows: {stats['labeled_rows']}")


if __name__ == "__main__":
    main()
