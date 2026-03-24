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
from spanlab.paths import default_silver_label_path, default_span_ready_path
from spanlab.silver import assign_silver_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct silver span labels from heuristic support rules.")
    parser.add_argument("--span_ready_path", type=Path, default=default_span_ready_path())
    parser.add_argument("--span_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--summary_path", type=Path, default=None)
    parser.add_argument("--positive_threshold", type=float, default=0.65)
    parser.add_argument("--negative_threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_rows = {row["sample_id"]: row for row in read_jsonl(args.span_ready_path)}
    span_rows = read_jsonl(args.span_path)

    route_name = args.span_path.stem
    output_path = args.output_path or default_silver_label_path(route_name)
    summary_path = args.summary_path or output_path.with_suffix(".summary.json")

    labeled_rows = []
    stats = Counter()
    for span_row in span_rows:
        sample_row = sample_rows[span_row["sample_id"]]
        labeled = assign_silver_label(
            span_row,
            sample_row,
            negative_threshold=args.negative_threshold,
            positive_threshold=args.positive_threshold,
        )
        labeled_rows.append(labeled)
        stats["total_spans"] += 1
        stats[f"decision_{labeled['silver_decision']}"] += 1
        if labeled["silver_label"] is not None:
            stats[f"label_{labeled['silver_label']}"] += 1
        stats[f"span_type_{labeled['span_type']}"] += 1

    write_jsonl(output_path, labeled_rows)
    dump_json(
        summary_path,
        {
            "span_ready_path": str(args.span_ready_path),
            "span_path": str(args.span_path),
            "output_path": str(output_path),
            "positive_threshold": args.positive_threshold,
            "negative_threshold": args.negative_threshold,
            **stats,
        },
    )
    print(f"Saved silver labels to: {output_path}")
    print(
        "Labeled spans: "
        f"pos={stats.get('label_1', 0)} neg={stats.get('label_0', 0)} skipped={stats.get('decision_skip', 0)}"
    )


if __name__ == "__main__":
    main()
