from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?")
EVIDENCE_RE = re.compile(r"D\d+:\d+")

NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "cannot",
    "can't",
    "wont",
    "won't",
    "avoid",
    "stop",
    "stopped",
    "cancel",
    "cancelled",
    "changed",
    "moved",
    "new",
    "instead",
    "replaced",
    "deprecated",
    "blocked",
    "injury",
    "injured",
    "broken",
    "crash",
    "risk",
    "unsafe",
    "hold",
}


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\x00", " ").split())


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def estimate_tokens(text: str) -> int:
    """Cheap token estimate good enough for comparing prompt budgets."""

    if not text:
        return 0
    words = len(tokenize(text))
    chars = len(text)
    return max(1, int(max(words * 1.25, chars / 4.0)))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def overlap_coeff(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    denom = min(len(sa), len(sb))
    if denom == 0:
        return 0.0
    return len(sa & sb) / denom


def cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def extract_evidence_ids(values: Sequence[object]) -> List[str]:
    ids: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        ids.extend(EVIDENCE_RE.findall(text))
    return ids


def contains_risk_terms(text: str) -> bool:
    toks = set(tokenize(text))
    return bool(toks & NEGATION_TERMS)


def truncate_to_token_budget(text: str, budget: int) -> str:
    if estimate_tokens(text) <= budget:
        return text
    words = text.split()
    if not words:
        return text
    # Binary search keeps this deterministic without external tokenizers.
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = " ".join(words[:mid])
        if estimate_tokens(candidate) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo]).rstrip() + " ..."

