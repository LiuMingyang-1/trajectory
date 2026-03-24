#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.dependencies import load_spacy_model
from spanlab.io_utils import dump_json, read_jsonl, write_jsonl
from spanlab.paths import default_spacy_span_path, default_span_ready_path
from spanlab.spans import build_spacy_spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spaCy-based span candidates.")
    parser.add_argument("--input_path", type=Path, default=default_span_ready_path())
    parser.add_argument("--output_path", type=Path, default=default_spacy_span_path())
    parser.add_argument("--summary_path", type=Path, default=None)
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=12)
    parser.add_argument("--disable_entities", action="store_true")
    parser.add_argument("--disable_noun_chunks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    nlp = load_spacy_model(args.spacy_model)

    candidates = []
    stats = Counter()
    for row in rows:
        built, row_stats = build_spacy_spans(
            row,
            nlp=nlp,
            include_entities=not args.disable_entities,
            include_noun_chunks=not args.disable_noun_chunks,
            max_tokens=args.max_tokens,
        )
        candidates.extend(built)
        stats["samples_total"] += 1
        for key, value in row_stats.items():
            stats[key] += value

    write_jsonl(args.output_path, candidates)
    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    dump_json(
        summary_path,
        {
            "input_path": str(args.input_path),
            "output_path": str(args.output_path),
            "spacy_model": args.spacy_model,
            "max_tokens": args.max_tokens,
            **stats,
        },
    )
    print(f"Saved spaCy spans to: {args.output_path}")
    print(f"Total spans: {len(candidates)}")


if __name__ == "__main__":
    main()
