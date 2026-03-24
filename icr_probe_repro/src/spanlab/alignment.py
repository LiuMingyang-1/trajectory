from typing import Any, Dict, List, Optional


USABLE_LAYERS = 27


def build_sample_id(index: int, candidate_index: int) -> str:
    return f"{index}:{candidate_index}"


def retokenize_response(tokenizer: Any, response: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded["input_ids"])
    offsets = [list(pair) for pair in encoded["offset_mapping"]]

    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
        offsets = offsets[:max_tokens]

    token_texts: List[str] = []
    fallback_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    for token_id, (start, end), fallback in zip(token_ids, offsets, fallback_tokens):
        if end > start:
            token_texts.append(response[start:end])
        else:
            token_texts.append(str(fallback))

    return {
        "response_token_ids": [int(token_id) for token_id in token_ids],
        "response_token_texts": token_texts,
        "response_offsets": offsets,
        "num_response_tokens_retokenized": len(token_ids),
    }


def build_alignment_report(icr_record: Dict[str, Any], retokenized: Dict[str, Any]) -> Dict[str, Any]:
    icr_scores = icr_record["icr_scores"]
    icr_token_count = len(icr_scores[0]) if icr_scores else 0
    expected_tokens = int(icr_record.get("num_response_tokens", -1))
    retokenized_tokens = int(retokenized["num_response_tokens_retokenized"])

    checks = {
        "matches_num_response_tokens": expected_tokens == retokenized_tokens,
        "matches_icr_token_count": icr_token_count == retokenized_tokens,
        "matches_between_original_fields": expected_tokens == icr_token_count,
    }
    errors = [name for name, value in checks.items() if not value]

    return {
        "alignment_ok": not errors,
        "alignment_checks": checks,
        "alignment_errors": errors,
        "icr_token_count": icr_token_count,
    }


def prepare_span_ready_record(
    icr_record: Dict[str, Any],
    qa_record: Dict[str, Any],
    tokenizer: Any,
    usable_layers: int = USABLE_LAYERS,
) -> Dict[str, Any]:
    if len(icr_record["icr_scores"]) < usable_layers:
        raise ValueError("ICR record has fewer layers than expected.")

    retokenized = retokenize_response(
        tokenizer=tokenizer,
        response=icr_record["response"],
        max_tokens=int(icr_record.get("num_response_tokens", 0)) or None,
    )
    report = build_alignment_report(icr_record, retokenized)

    sample_id = build_sample_id(int(icr_record["index"]), int(icr_record.get("candidate_index", 0)))

    return {
        "sample_id": sample_id,
        "source_sample_index": int(icr_record["index"]),
        "candidate_index": int(icr_record.get("candidate_index", 0)),
        "task": icr_record.get("task"),
        "pairing": icr_record.get("pairing"),
        "prompt": icr_record.get("prompt"),
        "question": qa_record.get("question"),
        "knowledge": qa_record.get("knowledge"),
        "response": icr_record.get("response"),
        "response_type": icr_record.get("response_type"),
        "sample_label": int(icr_record["label"]),
        "hallucination": qa_record.get("hallucination"),
        "model_name_or_path": icr_record.get("model_name_or_path"),
        "core_positions": icr_record.get("core_positions"),
        "num_layers_raw": int(icr_record.get("num_layers", len(icr_record["icr_scores"]))),
        "num_layers_usable": usable_layers,
        "icr_scores": icr_record["icr_scores"][:usable_layers],
        "num_response_tokens": int(icr_record.get("num_response_tokens", 0)),
        **retokenized,
        **report,
    }
