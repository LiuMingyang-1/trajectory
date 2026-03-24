from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


def token_char_bounds(offsets: List[List[int]], token_start: int, token_end: int) -> Tuple[int, int]:
    char_start = offsets[token_start][0]
    char_end = offsets[token_end - 1][1]
    return int(char_start), int(char_end)


def map_char_span_to_token_span(offsets: List[List[int]], char_start: int, char_end: int) -> Optional[Tuple[int, int]]:
    overlapping = [
        index
        for index, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > char_start and tok_start < char_end
    ]
    if not overlapping:
        return None
    return min(overlapping), max(overlapping) + 1


def _base_span_record(sample_row: Dict[str, Any], route: str, span_type: str, token_start: int, token_end: int) -> Dict[str, Any]:
    offsets = sample_row["response_offsets"]
    char_start, char_end = token_char_bounds(offsets, token_start, token_end)
    span_text = sample_row["response"][char_start:char_end]
    return {
        "sample_id": sample_row["sample_id"],
        "source_sample_index": sample_row["source_sample_index"],
        "candidate_index": sample_row["candidate_index"],
        "sample_label": sample_row["sample_label"],
        "route": route,
        "span_type": span_type,
        "token_start": token_start,
        "token_end": token_end,
        "char_start": char_start,
        "char_end": char_end,
        "token_char_start": char_start,
        "token_char_end": char_end,
        "span_len_tokens": token_end - token_start,
        "span_text": span_text,
    }


def build_tokenizer_windows(sample_row: Dict[str, Any], window_sizes: Iterable[int]) -> List[Dict[str, Any]]:
    if not sample_row.get("alignment_ok"):
        return []

    n_tokens = len(sample_row["response_token_ids"])
    rows: List[Dict[str, Any]] = []
    for window_size in window_sizes:
        if window_size <= 0 or window_size > n_tokens:
            continue
        for token_start in range(0, n_tokens - window_size + 1):
            token_end = token_start + window_size
            row = _base_span_record(sample_row, "tokenizer_window", "window", token_start, token_end)
            if not row["span_text"].strip():
                continue
            row["window_size"] = int(window_size)
            row["span_id"] = (
                f"{sample_row['sample_id']}::tokenizer_window::{window_size}::{token_start}-{token_end}"
            )
            rows.append(row)
    return rows


def build_spacy_spans(
    sample_row: Dict[str, Any],
    nlp: Any,
    include_entities: bool = True,
    include_noun_chunks: bool = True,
    max_tokens: int = 12,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not sample_row.get("alignment_ok"):
        return [], {"skipped_unaligned": 1}

    doc = nlp(sample_row["response"])
    offsets = sample_row["response_offsets"]
    rows: List[Dict[str, Any]] = []
    stats = Counter()
    seen = set()

    span_specs = []
    if include_entities:
        span_specs.extend((span.start_char, span.end_char, span.text, "entity") for span in doc.ents)
    if include_noun_chunks:
        span_specs.extend((span.start_char, span.end_char, span.text, "noun_chunk") for span in doc.noun_chunks)

    for char_start, char_end, _, span_type in span_specs:
        if char_end <= char_start:
            stats["empty_spans"] += 1
            continue
        token_span = map_char_span_to_token_span(offsets, char_start, char_end)
        if token_span is None:
            stats["unmapped_spans"] += 1
            continue
        token_start, token_end = token_span
        if token_end - token_start > max_tokens:
            stats["too_long_spans"] += 1
            continue

        row = _base_span_record(sample_row, "spacy_span", span_type, token_start, token_end)
        row["char_start"] = int(char_start)
        row["char_end"] = int(char_end)
        row["spacy_span_text"] = sample_row["response"][char_start:char_end]
        dedupe_key = (row["token_start"], row["token_end"], row["span_type"])
        if dedupe_key in seen:
            stats["duplicate_spans"] += 1
            continue
        seen.add(dedupe_key)
        row["span_id"] = f"{sample_row['sample_id']}::spacy_span::{span_type}::{token_start}-{token_end}"
        rows.append(row)
        stats[f"kept_{span_type}"] += 1

    return rows, dict(stats)
