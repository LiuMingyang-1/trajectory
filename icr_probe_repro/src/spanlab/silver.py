from typing import Any, Dict, List, Optional

from .text_utils import (
    content_words,
    looks_entity_like,
    normalize_text,
    numberish_tokens,
    overlap_ratio,
    stable_unique,
)


def score_span_support(span_row: Dict[str, Any], sample_row: Dict[str, Any]) -> Dict[str, Any]:
    span_text = span_row["span_text"]
    knowledge = sample_row.get("knowledge") or ""
    question = sample_row.get("question") or ""
    span_type = span_row.get("span_type") or ""

    span_norm = normalize_text(span_text)
    knowledge_norm = normalize_text(knowledge)
    question_terms = set(content_words(question))
    knowledge_terms = set(content_words(knowledge))

    span_terms = content_words(span_text)
    novel_terms = [term for term in span_terms if term not in question_terms]
    effective_terms = novel_terms if novel_terms else span_terms

    span_numbers = stable_unique(numberish_tokens(span_text))
    knowledge_numbers = set(numberish_tokens(knowledge))

    exact_match = bool(span_norm) and len(span_norm) >= 3 and span_norm in knowledge_norm
    term_overlap = overlap_ratio(effective_terms, knowledge_terms)
    all_terms_supported = bool(effective_terms) and term_overlap == 1.0
    looks_like_entity = span_type == "entity" or looks_entity_like(span_text)
    missing_numbers = [item for item in span_numbers if item not in knowledge_numbers]
    supported_numbers = [item for item in span_numbers if item in knowledge_numbers]

    support_score = 0.0
    unsupported_score = 0.0
    rules: List[str] = []

    if exact_match:
        support_score += 0.65
        rules.append("exact_knowledge_match")
    if all_terms_supported:
        support_score += 0.3
        rules.append("all_content_words_supported")
    elif term_overlap >= 0.67:
        support_score += 0.2
        rules.append("most_content_words_supported")
    elif effective_terms and term_overlap <= 0.25:
        unsupported_score += 0.3
        rules.append("low_lexical_support")

    if span_numbers:
        if missing_numbers:
            unsupported_score += 0.55
            rules.append("missing_numeric_support")
        if supported_numbers and not missing_numbers:
            support_score += 0.25
            rules.append("numeric_support_present")

    if looks_like_entity:
        if exact_match or term_overlap >= 0.67:
            support_score += 0.15
            rules.append("entity_grounded")
        elif effective_terms:
            unsupported_score += 0.2
            rules.append("entity_not_grounded")

    if not effective_terms and not span_numbers and not exact_match:
        rules.append("too_little_signal")

    support_score = min(1.0, support_score)
    unsupported_score = min(1.0, unsupported_score + max(0.0, 0.35 - support_score))

    return {
        "exact_match": exact_match,
        "span_terms": span_terms,
        "effective_terms": effective_terms,
        "term_overlap": term_overlap,
        "span_numbers": span_numbers,
        "supported_numbers": supported_numbers,
        "missing_numbers": missing_numbers,
        "looks_like_entity": looks_like_entity,
        "support_score": round(float(support_score), 6),
        "unsupported_score": round(float(unsupported_score), 6),
        "rules": stable_unique(rules),
    }


def assign_silver_label(
    span_row: Dict[str, Any],
    sample_row: Dict[str, Any],
    negative_threshold: float = 0.75,
    positive_threshold: float = 0.65,
) -> Dict[str, Any]:
    diagnostics = score_span_support(span_row, sample_row)
    sample_label = int(sample_row["sample_label"])
    effective_terms = diagnostics["effective_terms"]
    nonnumeric_terms = [term for term in effective_terms if not term.isdigit()]
    has_signal = bool(effective_terms or diagnostics["span_numbers"] or diagnostics["exact_match"])

    silver_label: Optional[int] = None
    confidence = 0.0
    decision = "skip"

    if diagnostics["span_numbers"] and not diagnostics["missing_numbers"] and not nonnumeric_terms:
        silver_label = 0
        confidence = max(diagnostics["support_score"], 0.8)
        decision = "negative"
    elif diagnostics["support_score"] >= negative_threshold:
        silver_label = 0
        confidence = diagnostics["support_score"]
        decision = "negative"
    elif sample_label == 1 and has_signal and diagnostics["unsupported_score"] >= positive_threshold:
        silver_label = 1
        confidence = diagnostics["unsupported_score"]
        decision = "positive"

    return {
        **span_row,
        "silver_label": silver_label,
        "silver_confidence": round(float(confidence), 6),
        "silver_decision": decision,
        "support_score": diagnostics["support_score"],
        "unsupported_score": diagnostics["unsupported_score"],
        "silver_rules": diagnostics["rules"],
        "silver_diagnostics": diagnostics,
    }
