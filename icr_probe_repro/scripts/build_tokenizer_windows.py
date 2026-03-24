#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.io_utils import dump_json, parse_int_list, read_jsonl, write_jsonl
from spanlab.paths import default_span_ready_path, default_tokenizer_window_path
from spanlab.spans import build_tokenizer_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tokenizer-window span candidates.")
    parser.add_argument("--input_path", type=Path, default=default_span_ready_path())
    parser.add_argument("--output_path", type=Path, default=default_tokenizer_window_path())
    parser.add_argument("--summary_path", type=Path, default=None)
    parser.add_argument("--window_sizes", type=str, default="1,2,3,4")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    window_sizes = parse_int_list(args.window_sizes)
    candidates = []
    stats = Counter()
    for row in rows:
        built = build_tokenizer_windows(row, window_sizes=window_sizes)
        candidates.extend(built)
        stats["samples_total"] += 1
        stats["samples_aligned"] += int(bool(row.get("alignment_ok")))
        stats["spans_total"] += len(built)
        for candidate in built:
            stats[f"window_size_{candidate['window_size']}"] += 1

    write_jsonl(args.output_path, candidates)
    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    dump_json(
        summary_path,
        {
            "input_path": str(args.input_path),
            "output_path": str(args.output_path),
            "window_sizes": window_sizes,
            **stats,
        },
    )
    print(f"Saved tokenizer windows to: {args.output_path}")
    print(f"Total spans: {stats['spans_total']}")


if __name__ == "__main__":
    main()
