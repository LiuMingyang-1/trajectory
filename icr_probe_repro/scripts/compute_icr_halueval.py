#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from icrprobe import ICRScore
from spanlab.paths import DEFAULT_HALUEVAL_QA_PATH, DEFAULT_ICR_OUTPUT_PATH


TASK_CHOICES = ["auto", "general", "qa", "dialogue", "summarization"]
PAIRING_CHOICES = ["auto", "both", "random", "right", "hallucinated", "single"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ICR scores for HaluEval samples.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_HALUEVAL_QA_PATH), help="HaluEval file path")
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_ICR_OUTPUT_PATH),
        help="Output JSONL path for ICR scores",
    )

    parser.add_argument("--task", type=str, default="auto", choices=TASK_CHOICES)
    parser.add_argument(
        "--pairing",
        type=str,
        default="auto",
        choices=PAIRING_CHOICES,
        help=(
            "How to use right/hallucinated pairs. "
            "auto: general->single, others->random (matching HaluEval evaluation); "
            "both: output two rows per sample."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prompt_key", type=str, default=None, help="Custom field name for prompt text")
    parser.add_argument("--response_key", type=str, default=None, help="Custom field name for response text")
    parser.add_argument("--label_key", type=str, default=None, help="Custom field name for hallucination label")
    parser.add_argument("--id_key", type=str, default=None, help="Custom field name for sample id")

    parser.add_argument("--split", type=str, default=None, help="Split key when JSON root is a dict")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_response_tokens", type=int, default=128)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend used by the model. Use eager when output_attentions is required.",
    )

    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--disable_chat_template", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)

    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "min"])
    parser.add_argument("--attention_uniform", action="store_true")
    parser.add_argument("--hidden_uniform", action="store_true")
    parser.add_argument("--use_induction_head", action="store_true")
    parser.add_argument("--skew_threshold", type=float, default=0)
    parser.add_argument("--entropy_threshold", type=float, default=1e5)

    return parser.parse_args()


def load_records(path: Path, split: Optional[str]) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")

    # 1) Try strict JSON first (list or dict with split)
    try:
        raw = json.loads(text)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if split is not None:
                if split not in raw:
                    raise KeyError(f"split '{split}' not found in JSON keys: {list(raw.keys())}")
                if not isinstance(raw[split], list):
                    raise TypeError(f"JSON split '{split}' is not a list")
                return raw[split]
            for k in ("data", "train", "validation", "test", "dev"):
                if k in raw and isinstance(raw[k], list):
                    return raw[k]
            raise TypeError("JSON is a dict but no list-like split found; pass --split")
    except json.JSONDecodeError:
        pass

    # 2) Fallback: JSONL (HaluEval data files use this layout)
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def pick_first_key(record: Dict[str, Any], candidates: Iterable[str], explicit_key: Optional[str]) -> Optional[str]:
    if explicit_key is not None:
        if explicit_key not in record:
            raise KeyError(f"Key '{explicit_key}' not found in record keys: {list(record.keys())}")
        return explicit_key

    for key in candidates:
        if key in record:
            return key

    return None


def build_prompt(tokenizer, prompt: str, use_chat_template: bool, system_prompt: Optional[str]) -> str:
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def tokenize_text(tokenizer, text: str, max_len: Optional[int] = None) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"][0]
    if max_len is not None:
        ids = ids[:max_len]
    return ids


def infer_halueval_task(record: Dict[str, Any], specified: str) -> str:
    if specified != "auto":
        return specified
    keys = set(record.keys())
    if {"user_query", "chatgpt_response"}.issubset(keys):
        return "general"
    if {"knowledge", "question", "right_answer", "hallucinated_answer"}.issubset(keys):
        return "qa"
    if {"knowledge", "question", "answer", "hallucination"}.issubset(keys):
        return "qa"
    if {"knowledge", "dialogue_history", "right_response", "hallucinated_response"}.issubset(keys):
        return "dialogue"
    if {"document", "right_summary", "hallucinated_summary"}.issubset(keys):
        return "summarization"
    raise ValueError(f"Cannot infer task from keys: {sorted(keys)}")


def normalize_binary_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "hallucinated", "hallucination"}:
        return 1
    if text in {"no", "n", "false", "0", "non-hallucinated", "non_hallucinated"}:
        return 0
    return None


def make_halueval_candidates(
    record: Dict[str, Any],
    task: str,
    pairing: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if pairing == "auto":
        pairing = "single" if task == "general" else "random"

    if task == "general":
        prompt = str(record["user_query"])
        response = str(record["chatgpt_response"])
        label = normalize_binary_label(record.get("hallucination"))
        return [{"prompt": prompt, "response": response, "label": label, "response_type": "chatgpt_response"}]

    if task == "qa":
        prompt = f"Knowledge:\n{record['knowledge']}\n\nQuestion:\n{record['question']}\n\nAnswer:"
        # HuggingFace HaluEval uses single-answer format: answer + hallucination label
        if "right_answer" not in record and "answer" in record:
            response = str(record["answer"])
            label = normalize_binary_label(record.get("hallucination"))
            return [{"prompt": prompt, "response": response, "label": label, "response_type": "answer"}]
        right = str(record["right_answer"])
        hallu = str(record["hallucinated_answer"])
    elif task == "dialogue":
        prompt = f"Knowledge:\n{record['knowledge']}\n\nDialogue History:\n{record['dialogue_history']}\n\nAssistant Response:"
        right = str(record["right_response"])
        hallu = str(record["hallucinated_response"])
    elif task == "summarization":
        prompt = f"Document:\n{record['document']}\n\nSummary:"
        right = str(record["right_summary"])
        hallu = str(record["hallucinated_summary"])
    else:
        raise ValueError(f"Unsupported task: {task}")

    both = [
        {"prompt": prompt, "response": right, "label": 0, "response_type": "right"},
        {"prompt": prompt, "response": hallu, "label": 1, "response_type": "hallucinated"},
    ]

    if pairing == "both":
        return both
    if pairing == "random":
        return [both[rng.randint(0, 1)]]
    if pairing == "right":
        return [both[0]]
    if pairing == "hallucinated":
        return [both[1]]

    raise ValueError(f"Pairing '{pairing}' is invalid for task '{task}'")


def make_custom_candidate(record: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    prompt_key = pick_first_key(record, ["question", "prompt", "query", "instruction", "input"], args.prompt_key)
    response_key = pick_first_key(record, ["answer", "response", "model_output", "output", "text"], args.response_key)
    label_key = pick_first_key(record, ["label", "hallucination", "is_hallucination", "binary_label"], args.label_key)

    if prompt_key is None or response_key is None:
        raise ValueError(
            "Custom mode requires detectable prompt/response keys. "
            "Please pass --prompt_key and --response_key explicitly."
        )

    label = normalize_binary_label(record.get(label_key)) if label_key is not None else None
    return [{
        "prompt": str(record[prompt_key]),
        "response": str(record[response_key]),
        "label": label,
        "response_type": response_key,
        "prompt_key": prompt_key,
        "response_key": response_key,
        "label_key": label_key,
    }]


def collect_stepwise_cache(
    model,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    device: str,
) -> Tuple[List[Any], List[Any]]:
    """Single forward pass over prompt+response, then split into step-by-step format
    expected by ICRScore. Equivalent to token-by-token KV-cache decoding
    (causal mask guarantees identical hidden states / attentions) but much faster."""
    prompt_len = prompt_ids.numel()
    full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(
            input_ids=full_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    # Convert single-pass outputs to the step-by-step list format ICRScore expects.
    # Step 0 (prompt): hidden [1, prompt_len, H], attn [1, heads, prompt_len, prompt_len]
    # Step i (response token i): hidden [1, 1, H], attn [1, heads, 1, prompt_len+i]
    hidden_states_steps: List[Any] = []
    attentions_steps: List[Any] = []

    # Step 0: prompt
    hs_prompt = tuple(h[:, :prompt_len, :] for h in out.hidden_states)
    attn_prompt = tuple(a[:, :, :prompt_len, :prompt_len] for a in out.attentions)
    hidden_states_steps.append(hs_prompt)
    attentions_steps.append(attn_prompt)

    # Steps 1..N: each response token
    for i in range(response_ids.numel()):
        pos = prompt_len + i
        hs_step = tuple(h[:, pos:pos+1, :] for h in out.hidden_states)
        attn_step = tuple(a[:, :, pos:pos+1, :pos+1] for a in out.attentions)
        hidden_states_steps.append(hs_step)
        attentions_steps.append(attn_step)

    return hidden_states_steps, attentions_steps


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    use_chat_template = args.use_chat_template and not args.disable_chat_template

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = args.device
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    records = load_records(Path(args.data_path), args.split)
    if args.start_index:
        records = records[args.start_index :]
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No records to process")

    inferred_task = infer_halueval_task(records[0], args.task)
    id_key = pick_first_key(records[0], ["id", "ID", "sample_id", "idx"], args.id_key)

    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            task = infer_halueval_task(rec, inferred_task)

            if task in {"general", "qa", "dialogue", "summarization"}:
                candidates = make_halueval_candidates(rec, task, args.pairing, rng)
            else:
                candidates = make_custom_candidate(rec, args)

            for cand_idx, cand in enumerate(candidates):
                prompt = cand["prompt"]
                response = cand["response"]

                full_prompt = build_prompt(tokenizer, prompt, use_chat_template=use_chat_template, system_prompt=args.system_prompt)
                prompt_ids = tokenize_text(tokenizer, full_prompt)
                response_ids = tokenize_text(tokenizer, response, max_len=args.max_response_tokens)

                if response_ids.numel() == 0:
                    skipped += 1
                    continue

                hidden_states, attentions = collect_stepwise_cache(
                    model=model,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    device=device,
                )

                input_len = int(prompt_ids.numel())
                core_positions = {
                    "user_prompt_start": 0,
                    "user_prompt_end": input_len,
                    "response_start": input_len,
                }

                icr_calculator = ICRScore(
                    hidden_states=hidden_states,
                    attentions=attentions,
                    skew_threshold=args.skew_threshold,
                    entropy_threshold=args.entropy_threshold,
                    core_positions=core_positions,
                    icr_device=device,
                )
                icr_scores, top_p_mean = icr_calculator.compute_icr(
                    top_k=args.top_k,
                    top_p=args.top_p,
                    pooling=args.pooling,
                    attention_uniform=args.attention_uniform,
                    hidden_uniform=args.hidden_uniform,
                    use_induction_head=args.use_induction_head,
                )

                row = {
                    "index": i + args.start_index,
                    "candidate_index": cand_idx,
                    "task": task,
                    "pairing": args.pairing,
                    "prompt": prompt,
                    "response": response,
                    "response_type": cand.get("response_type"),
                    "label": cand.get("label"),
                    "model_name_or_path": args.model_name_or_path,
                    "icr_scores": icr_scores,
                    "top_p_mean": float(top_p_mean),
                    "num_layers": len(icr_scores),
                    "num_response_tokens": int(response_ids.numel()),
                    "core_positions": core_positions,
                }
                if id_key is not None and id_key in rec:
                    row["id"] = rec[id_key]

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            if (i + 1) % 10 == 0:
                print(f"Processed samples: {i + 1}/{len(records)}, written rows: {written}, skipped: {skipped}")

    print(f"Done. Saved ICR scores to {output_path}. Written rows: {written}, skipped rows: {skipped}")


if __name__ == "__main__":
    main()
