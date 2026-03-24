#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.alignment import prepare_span_ready_record
from spanlab.dependencies import require_transformers
from spanlab.io_utils import dump_json, load_json_or_jsonl, read_jsonl, write_jsonl
from spanlab.paths import (
    DEFAULT_HALUEVAL_QA_PATH,
    DEFAULT_ICR_OUTPUT_PATH,
    default_alignment_summary_path,
    default_span_ready_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover response token metadata for span-level ICR experiments.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_ICR_OUTPUT_PATH)
    parser.add_argument("--qa_data_path", type=Path, default=DEFAULT_HALUEVAL_QA_PATH)
    parser.add_argument("--output_path", type=Path, default=default_span_ready_path())
    parser.add_argument("--summary_path", type=Path, default=default_alignment_summary_path())
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    icr_rows = read_jsonl(args.input_path)
    qa_rows = load_json_or_jsonl(args.qa_data_path)
    if args.max_samples is not None:
        icr_rows = icr_rows[: args.max_samples]

    if not icr_rows:
        raise ValueError("No ICR rows found.")

    model_name_or_path = args.model_name_or_path or icr_rows[0].get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("Could not infer tokenizer model name from ICR file. Pass --model_name_or_path explicitly.")

    transformers = require_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    prepared_rows = []
    mismatch_examples = []
    aligned = 0
    for icr_row in icr_rows:
        qa_row = qa_rows[int(icr_row["index"])]
        prepared = prepare_span_ready_record(icr_row, qa_row, tokenizer)
        prepared_rows.append(prepared)
        if prepared["alignment_ok"]:
            aligned += 1
        elif len(mismatch_examples) < 5:
            mismatch_examples.append(
                {
                    "sample_id": prepared["sample_id"],
                    "source_sample_index": prepared["source_sample_index"],
                    "num_response_tokens": prepared["num_response_tokens"],
                    "num_response_tokens_retokenized": prepared["num_response_tokens_retokenized"],
                    "icr_token_count": prepared["icr_token_count"],
                    "alignment_errors": prepared["alignment_errors"],
                    "response": prepared["response"],
                }
            )

    write_jsonl(args.output_path, prepared_rows)
    summary = {
        "input_path": str(args.input_path),
        "qa_data_path": str(args.qa_data_path),
        "output_path": str(args.output_path),
        "model_name_or_path": model_name_or_path,
        "total_rows": len(prepared_rows),
        "aligned_rows": aligned,
        "misaligned_rows": len(prepared_rows) - aligned,
        "alignment_rate": aligned / max(len(prepared_rows), 1),
        "mismatch_examples": mismatch_examples,
    }
    dump_json(args.summary_path, summary)

    print(f"Saved span-ready data to: {args.output_path}")
    print(f"Alignment summary saved to: {args.summary_path}")
    print(f"Aligned rows: {aligned}/{len(prepared_rows)}")


if __name__ == "__main__":
    main()
