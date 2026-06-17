from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List


def actmem_style_metrics(memory_metrics: Dict, answer_metrics: Dict, judge: Dict) -> Dict[str, float]:
    """Paper-aligned metrics for ActMem-like QA.

    The ActMem-style view separates memory evidence retrieval from answer
    correctness. With a real LLM judge, `qa_accuracy` should be read as the
    primary answer metric. With the heuristic judge, it is a smoke-test proxy.
    """

    return {
        "retrieval_accuracy": float(memory_metrics.get("evidence_recall", 0.0)),
        "retrieval_precision": float(memory_metrics.get("evidence_precision", 0.0)),
        "qa_accuracy": float(judge.get("correct", 0.0)),
        "qa_score": float(judge.get("score", 0.0)),
        "answer_f1": float(answer_metrics.get("token_f1", 0.0)),
        "answer_rouge_l": float(answer_metrics.get("rouge_l_f1", 0.0)),
        "token_cost": float(memory_metrics.get("token_estimate", 0.0)),
    }


def locomo_style_metrics(memory_metrics: Dict, answer_metrics: Dict, judge: Dict) -> Dict[str, float]:
    """LoCoMo-like long-term-memory QA metrics.

    Full LoCoMo evaluation often uses answer quality with a model judge plus
    evidence diagnostics. This function keeps both in one row.
    """

    return {
        "evidence_recall": float(memory_metrics.get("evidence_recall", 0.0)),
        "evidence_precision": float(memory_metrics.get("evidence_precision", 0.0)),
        "qa_accuracy": float(judge.get("correct", 0.0)),
        "qa_score": float(judge.get("score", 0.0)),
        "answer_f1": float(answer_metrics.get("token_f1", 0.0)),
        "answer_rouge_l": float(answer_metrics.get("rouge_l_f1", 0.0)),
        "token_cost": float(memory_metrics.get("token_estimate", 0.0)),
    }


def aggregate_paper_metrics(rows: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["result"]["method"]].append(row.get("paper_metrics", {}))

    summary: Dict[str, Dict[str, float]] = {}
    for method, items in grouped.items():
        keys = sorted({k for item in items for k in item})
        out = {"n": float(len(items))}
        for key in keys:
            out[key] = mean(float(item.get(key, 0.0)) for item in items)
        summary[method] = out
    return summary


def stale_style_dimension_metrics(rows: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate STALE-like SR/PR/IPA metrics by method.

    Expected dimensions:
    - SR: status recognition, detect old memory is invalidated.
    - PR: premise resistance, reject a question that assumes stale memory.
    - IPA: implicit preference/action, answer with fresh memory in downstream use.
    """

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["result"]["method"]].append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for method, items in grouped.items():
        out = {"n": float(len(items))}
        for dim in ["SR", "PR", "IPA"]:
            dim_items = [x for x in items if x["case"].get("probe_dimension") == dim]
            if not dim_items:
                continue
            out[f"{dim.lower()}_accuracy"] = mean(
                float(x["stale_metrics"].get("stale_probe_accuracy", 0.0)) for x in dim_items
            )
            out[f"{dim.lower()}_stale_answer_leak"] = mean(
                float(x["stale_metrics"].get("stale_answer_leak", 0.0)) for x in dim_items
            )
        out["stale_overall_accuracy"] = mean(
            float(x["stale_metrics"].get("stale_probe_accuracy", 0.0)) for x in items
        )
        summary[method] = out
    return summary
