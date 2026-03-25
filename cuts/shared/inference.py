"""Extract per-layer logit entropy from model hidden states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.entropy import compute_all_layer_entropies, entropy_summary_stats
from shared.paths import ENTROPY_JSONL, ICR_INPUT_JSONL


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for entropy extraction."""
    parser = argparse.ArgumentParser(description="Extract per-layer entropy from model hidden states.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--icr_input_path", type=str, default=str(ICR_INPUT_JSONL))
    parser.add_argument("--output_path", type=str, default=str(ENTROPY_JSONL))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max_response_tokens", type=int, default=128)
    parser.add_argument("--batch_report_interval", type=int, default=10)
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Keep a partial output file even if some records fail during extraction.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    """Map a user-facing dtype name to a torch dtype."""
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_chat_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    """Apply the tokenizer chat template when available."""
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_text(tokenizer: AutoTokenizer, text: str, max_len: Optional[int] = None) -> torch.Tensor:
    """Tokenize a string without adding special tokens."""
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][0]
    if max_len is not None:
        input_ids = input_ids[:max_len]
    return input_ids


def load_icr_records(
    path: Path,
    start_index: int = 0,
    max_samples: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Load ICR JSONL rows and optionally slice them."""
    if not path.exists():
        raise FileNotFoundError(f"ICR input file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))

    if start_index:
        records = records[start_index:]
    if max_samples is not None:
        records = records[:max_samples]
    return records


def extract_entropy_for_record(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict[str, Any],
    device: str,
    max_response_tokens: int,
) -> dict[str, Any]:
    """Run a forward pass and compute per-layer entropy for a single row."""
    prompt = str(record["prompt"])
    response = str(record["response"])

    full_prompt = build_chat_prompt(tokenizer, prompt)
    prompt_ids = tokenize_text(tokenizer, full_prompt)
    response_ids = tokenize_text(tokenizer, response, max_len=max_response_tokens)

    full_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0).to(device)
    prompt_len = int(prompt_ids.numel())

    with torch.inference_mode():
        outputs = model(
            input_ids=full_ids,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )

    lm_head = model.get_output_embeddings()
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise ValueError("Model does not expose output embeddings required for entropy projection.")

    entropy_matrix = compute_all_layer_entropies(
        hidden_states=outputs.hidden_states,
        lm_head_weight=lm_head.weight,
        lm_head_bias=getattr(lm_head, "bias", None),
        response_start=prompt_len,
        device=device,
    )

    result = {
        "index": record["index"],
        "candidate_index": record.get("candidate_index", 0),
        "label": record["label"],
        "entropy_scores": entropy_matrix.tolist(),
        "num_response_tokens": int(response_ids.numel()),
        "num_layers": int(entropy_matrix.shape[0]),
    }

    del full_ids
    del prompt_ids
    del response_ids
    del outputs
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return result


def load_model(
    model_name_or_path: str,
    torch_dtype: torch.dtype,
    device: str,
) -> AutoModelForCausalLM:
    """Load the causal LM, preferring SDPA when the local stack supports it."""
    common_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": None,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation="sdpa",
            **common_kwargs,
        )
    except (TypeError, ValueError):
        print("Model loader does not support attn_implementation; retrying without it.")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **common_kwargs)

    return model.to(device)


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    dtype = dtype_from_name(args.dtype)
    if args.device == "cpu" and dtype == torch.float16:
        print("float16 is not reliable on CPU, switching to float32")
        dtype = torch.float32

    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_name_or_path}...")
    model = load_model(args.model_name_or_path, dtype, args.device)
    model.eval()

    print(f"Loading ICR records from {args.icr_input_path}...")
    records = load_icr_records(Path(args.icr_input_path), args.start_index, args.max_samples)
    print(f"Processing {len(records)} records...")

    written = 0
    skipped = 0
    last_result: Optional[dict[str, Any]] = None
    skipped_examples: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as fout:
        for row_idx, record in enumerate(records, start=1):
            try:
                result = extract_entropy_for_record(
                    model=model,
                    tokenizer=tokenizer,
                    record=record,
                    device=args.device,
                    max_response_tokens=args.max_response_tokens,
                )
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1
                last_result = result
            except Exception as exc:
                record_index = int(record.get("index", row_idx - 1))
                print(f"  WARNING: skipped record {record_index}: {exc}")
                skipped += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(
                        {
                            "index": record_index,
                            "candidate_index": int(record.get("candidate_index", 0)),
                            "error": str(exc),
                        }
                    )

            if args.batch_report_interval > 0 and row_idx % args.batch_report_interval == 0:
                print(f"  Processed {row_idx}/{len(records)}, written={written}, skipped={skipped}")
                if last_result is not None:
                    stats = entropy_summary_stats(np.asarray(last_result["entropy_scores"], dtype=np.float32))
                    print(f"    Last sample entropy stats: {stats}")

    if skipped and not args.allow_partial:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass

        print(
            "ERROR: entropy extraction did not finish cleanly. "
            f"Skipped {skipped} of {len(records)} records, and the partial output was removed."
        )
        if skipped_examples:
            print("First failed records:")
            for example in skipped_examples:
                print(
                    "  "
                    f"index={example['index']} candidate_index={example['candidate_index']} "
                    f"error={example['error']}"
                )
        print("Re-run after fixing the failing records, or pass `--allow_partial` to keep a partial file.")
        sys.exit(1)

    print(f"Done. Saved entropy scores to {output_path}")
    print(f"Written: {written}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
