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
    parser.add_argument("--batch_size", type=int, default=8)
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

    del full_ids, prompt_ids, response_ids, outputs
    return result


def _prepare_batch(
    tokenizer: AutoTokenizer,
    records: list[dict[str, Any]],
    max_response_tokens: int,
) -> tuple[list[torch.Tensor], list[int]]:
    """Tokenize a batch of records, return per-sample full_ids and prompt_lens."""
    all_ids: list[torch.Tensor] = []
    prompt_lens: list[int] = []
    for record in records:
        full_prompt = build_chat_prompt(tokenizer, str(record["prompt"]))
        prompt_ids = tokenize_text(tokenizer, full_prompt)
        response_ids = tokenize_text(tokenizer, str(record["response"]), max_len=max_response_tokens)
        full_ids = torch.cat([prompt_ids, response_ids], dim=0)
        all_ids.append(full_ids)
        prompt_lens.append(int(prompt_ids.numel()))
    return all_ids, prompt_lens


def _pad_and_batch(
    all_ids: list[torch.Tensor],
    pad_token_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a list of 1-D id tensors into a batch with attention mask."""
    max_len = max(ids.numel() for ids in all_ids)
    batch_ids = torch.full((len(all_ids), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)
    for i, ids in enumerate(all_ids):
        length = ids.numel()
        # Left-pad: place content at the right end
        batch_ids[i, max_len - length:] = ids
        attention_mask[i, max_len - length:] = 1
    return batch_ids.to(device), attention_mask.to(device)


def extract_entropy_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict[str, Any]],
    device: str,
    max_response_tokens: int,
) -> list[dict[str, Any]]:
    """Run a batched forward pass and compute per-layer entropy for each record."""
    all_ids, prompt_lens = _prepare_batch(tokenizer, records, max_response_tokens)
    seq_lens = [ids.numel() for ids in all_ids]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_ids, attention_mask = _pad_and_batch(all_ids, pad_token_id, device)
    max_len = batch_ids.shape[1]

    with torch.inference_mode():
        outputs = model(
            input_ids=batch_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )

    lm_head = model.get_output_embeddings()
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise ValueError("Model does not expose output embeddings required for entropy projection.")

    lm_weight = lm_head.weight
    lm_bias = getattr(lm_head, "bias", None)

    results: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        # Because of left-padding, content starts at (max_len - seq_lens[i])
        content_start = max_len - seq_lens[i]
        response_start_in_padded = content_start + prompt_lens[i]

        # Extract per-sample hidden states (single sample, no padding region)
        sample_hidden_states = tuple(
            hs[i:i+1, content_start:] for hs in outputs.hidden_states
        )

        entropy_matrix = compute_all_layer_entropies(
            hidden_states=sample_hidden_states,
            lm_head_weight=lm_weight,
            lm_head_bias=lm_bias,
            response_start=prompt_lens[i],
            device=device,
        )

        n_response_tokens = seq_lens[i] - prompt_lens[i]
        results.append({
            "index": record["index"],
            "candidate_index": record.get("candidate_index", 0),
            "label": record["label"],
            "entropy_scores": entropy_matrix.tolist(),
            "num_response_tokens": n_response_tokens,
            "num_layers": int(entropy_matrix.shape[0]),
        })

    del batch_ids, attention_mask, outputs
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return results


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
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {args.model_name_or_path}...")
    model = load_model(args.model_name_or_path, dtype, args.device)
    model.eval()

    print(f"Loading ICR records from {args.icr_input_path}...")
    records = load_icr_records(Path(args.icr_input_path), args.start_index, args.max_samples)
    print(f"Processing {len(records)} records (batch_size={args.batch_size})...")

    # Sort by total sequence length to minimize padding waste, track original order
    indexed_records = list(enumerate(records))
    indexed_records.sort(
        key=lambda ir: len(str(ir[1].get("prompt", ""))) + len(str(ir[1].get("response", ""))),
    )

    written = 0
    skipped = 0
    last_result: Optional[dict[str, Any]] = None
    skipped_examples: list[dict[str, Any]] = []

    # Collect results with original indices for ordered output
    all_results: list[tuple[int, dict[str, Any]]] = []

    total = len(indexed_records)
    batch_size = args.batch_size

    for batch_start in range(0, total, batch_size):
        batch_indexed = indexed_records[batch_start:batch_start + batch_size]
        batch_orig_indices = [idx for idx, _ in batch_indexed]
        batch_records = [rec for _, rec in batch_indexed]

        try:
            batch_results = extract_entropy_batch(
                model=model,
                tokenizer=tokenizer,
                records=batch_records,
                device=args.device,
                max_response_tokens=args.max_response_tokens,
            )
            for orig_idx, result in zip(batch_orig_indices, batch_results):
                all_results.append((orig_idx, result))
                written += 1
                last_result = result
        except Exception as exc:
            # Fallback: try each record individually
            for orig_idx, record in zip(batch_orig_indices, batch_records):
                try:
                    result = extract_entropy_for_record(
                        model=model,
                        tokenizer=tokenizer,
                        record=record,
                        device=args.device,
                        max_response_tokens=args.max_response_tokens,
                    )
                    all_results.append((orig_idx, result))
                    written += 1
                    last_result = result
                except Exception as inner_exc:
                    record_index = int(record.get("index", orig_idx))
                    print(f"  WARNING: skipped record {record_index}: {inner_exc}")
                    skipped += 1
                    if len(skipped_examples) < 10:
                        skipped_examples.append({
                            "index": record_index,
                            "candidate_index": int(record.get("candidate_index", 0)),
                            "error": str(inner_exc),
                        })

        processed = min(batch_start + batch_size, total)
        if args.batch_report_interval > 0 and processed % (args.batch_report_interval * batch_size) < batch_size:
            print(f"  Processed {processed}/{total}, written={written}, skipped={skipped}")
            if last_result is not None:
                stats = entropy_summary_stats(np.asarray(last_result["entropy_scores"], dtype=np.float32))
                print(f"    Last sample entropy stats: {stats}")

    # Write in original order
    all_results.sort(key=lambda x: x[0])
    with output_path.open("w", encoding="utf-8") as fout:
        for _, result in all_results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

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
