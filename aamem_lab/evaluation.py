from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .baselines import MemoryMethod
from .metrics import CaseMetrics, aggregate_by_method, score_result
from .schemas import ExperimentCase
from .serialization import method_result_to_dict


def run_methods_on_cases(
    cases: Sequence[ExperimentCase],
    methods: Sequence[MemoryMethod],
) -> tuple[List[Dict], List[CaseMetrics]]:
    rows: List[Dict] = []
    metrics: List[CaseMetrics] = []
    for case in cases:
        for method in methods:
            result = method.run(case)
            metric = score_result(case, result)
            metrics.append(metric)
            rows.append(
                {
                    "case": {
                        "id": case.query.id,
                        "dataset": case.query.source_dataset,
                        "sample_id": case.query.sample_id,
                        "query": case.query.query,
                        "answer": case.query.answer,
                        "gold_memory_ids": case.query.gold_memory_ids,
                        "category": case.query.category,
                    },
                    "result": method_result_to_dict(result, include_context=True),
                    "metrics": metric.__dict__,
                }
            )
    return rows, metrics


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: str | Path, metrics: Iterable[CaseMetrics]) -> Dict[str, Dict[str, float]]:
    summary = aggregate_by_method(metrics)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    if not summary:
        print("No metrics.")
        return
    headers = [
        "method",
        "n",
        "recall",
        "precision",
        "hit",
        "noise",
        "included",
        "tokens",
    ]
    print("\t".join(headers))
    for method, vals in summary.items():
        print(
            "\t".join(
                [
                    method,
                    f"{vals.get('n', 0):.0f}",
                    f"{vals.get('evidence_recall', 0):.3f}",
                    f"{vals.get('evidence_precision', 0):.3f}",
                    f"{vals.get('hit_rate', 0):.3f}",
                    f"{vals.get('noise_rate', 0):.3f}",
                    f"{vals.get('avg_included', 0):.1f}",
                    f"{vals.get('avg_tokens', 0):.0f}",
                ]
            )
        )

