from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List

from .text import estimate_tokens, tokenize


ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
PUNCT_RE = re.compile(r"[^A-Za-z0-9\s]")


@dataclass
class AnswerMetrics:
    exact_match: float
    contains_gold: float
    token_f1: float
    rouge_l_f1: float
    answer_tokens: int


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _f1_tokens(gold: str, pred: str) -> float:
    gold_toks = tokenize(normalize_answer(gold))
    pred_toks = tokenize(normalize_answer(pred))
    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(gold: str, pred: str) -> float:
    gold_toks = tokenize(normalize_answer(gold))
    pred_toks = tokenize(normalize_answer(pred))
    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0
    lcs = _lcs_len(gold_toks, pred_toks)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_toks)
    recall = lcs / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def score_answer(gold: str, pred: str) -> AnswerMetrics:
    ngold = normalize_answer(gold)
    npred = normalize_answer(pred)
    exact = 1.0 if ngold == npred and ngold else 0.0
    contains = 1.0 if ngold and (ngold in npred or npred in ngold) else 0.0
    return AnswerMetrics(
        exact_match=exact,
        contains_gold=contains,
        token_f1=_f1_tokens(gold, pred),
        rouge_l_f1=_rouge_l_f1(gold, pred),
        answer_tokens=estimate_tokens(pred),
    )


def aggregate_answer_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(rows)
    if not items:
        return {}
    keys = [
        "exact_match",
        "contains_gold",
        "token_f1",
        "rouge_l_f1",
        "answer_tokens",
        "latency_sec",
        "prompt_tokens_est",
        "completion_tokens_est",
    ]
    out: Dict[str, float] = {"n": float(len(items))}
    for key in keys:
        vals = [float(row.get(key, 0.0)) for row in items]
        out[f"avg_{key}"] = mean(vals)
    return out
