from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List

from .schemas import ExperimentCase, MethodResult


@dataclass
class CaseMetrics:
    case_id: str
    method: str
    gold_count: int
    included_count: int
    gold_hits: int
    evidence_recall: float
    evidence_precision: float
    hit: float
    noise_rate: float
    token_estimate: int


def score_result(case: ExperimentCase, result: MethodResult) -> CaseMetrics:
    gold = set(case.query.gold_memory_ids)
    included = set(result.included_memory_ids)
    hits = len(gold & included)
    included_count = len(included)
    gold_count = len(gold)
    recall = hits / gold_count if gold_count else 0.0
    precision = hits / included_count if included_count else 0.0
    noise_rate = (included_count - hits) / included_count if included_count else 0.0
    return CaseMetrics(
        case_id=case.query.id,
        method=result.method,
        gold_count=gold_count,
        included_count=included_count,
        gold_hits=hits,
        evidence_recall=recall,
        evidence_precision=precision,
        hit=1.0 if hits > 0 else 0.0,
        noise_rate=noise_rate,
        token_estimate=result.token_estimate,
    )


def aggregate(metrics: Iterable[CaseMetrics]) -> Dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {}
    return {
        "n": float(len(rows)),
        "evidence_recall": mean(r.evidence_recall for r in rows),
        "evidence_precision": mean(r.evidence_precision for r in rows),
        "hit_rate": mean(r.hit for r in rows),
        "noise_rate": mean(r.noise_rate for r in rows),
        "avg_included": mean(r.included_count for r in rows),
        "avg_tokens": mean(r.token_estimate for r in rows),
        "answerable_n": float(sum(1 for r in rows if r.gold_count > 0)),
    }


def aggregate_by_method(metrics: Iterable[CaseMetrics]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[CaseMetrics]] = {}
    for row in metrics:
        grouped.setdefault(row.method, []).append(row)
    return {method: aggregate(rows) for method, rows in grouped.items()}

