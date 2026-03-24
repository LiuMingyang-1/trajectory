import re
import string
from typing import Iterable, List, Sequence, Set


STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "will",
    "with",
    "which",
    "who",
    "whom",
    "whose",
    "what",
    "when",
    "where",
    "why",
    "how",
    "this",
    "these",
    "those",
    "first",
    "second",
    "third",
    "answer",
    "question",
}


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
NUMBERISH_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")


def normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"\s+", " ", re.sub(rf"[{re.escape(string.punctuation)}]", " ", lowered))
    return cleaned.strip()


def word_tokens(text: str) -> List[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def content_words(text: str) -> List[str]:
    tokens = word_tokens(text)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def numberish_tokens(text: str) -> List[str]:
    return [match.group(0) for match in NUMBERISH_RE.finditer(text)]


def looks_entity_like(text: str) -> bool:
    parts = [part for part in text.strip().split() if part]
    if not parts:
        return False
    capitalized = sum(part[:1].isupper() for part in parts if part[:1].isalpha())
    return capitalized >= 1 and capitalized >= max(1, len(parts) // 2)


def overlap_ratio(tokens: Sequence[str], support_set: Set[str]) -> float:
    if not tokens:
        return 0.0
    covered = sum(token in support_set for token in tokens)
    return covered / len(tokens)


def stable_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
