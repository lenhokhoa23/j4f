from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from .schemas import Candidate, GateDecision, MethodResult


def candidate_to_dict(candidate: Candidate) -> Dict[str, Any]:
    return {
        "memory_id": candidate.memory.id,
        "score": candidate.score,
        "rank": candidate.rank,
        "source": candidate.source,
        "expanded_from": candidate.expanded_from,
        "timestamp": candidate.memory.timestamp,
        "is_gold": candidate.memory.is_gold,
        "content_preview": candidate.memory.short(220),
    }


def gate_decision_to_dict(decision: GateDecision) -> Dict[str, Any]:
    return {
        "memory_id": decision.memory.id,
        "label": decision.label,
        "decision": decision.decision,
        "scores": decision.scores,
        "reason": decision.reason,
        "rank": decision.rank,
        "source": decision.source,
        "expanded_from": decision.expanded_from,
        "is_gold": decision.memory.is_gold,
        "content_preview": decision.memory.short(220),
    }


def method_result_to_dict(result: MethodResult, include_context: bool = False) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": result.method,
        "case_id": result.case_id,
        "included_memory_ids": result.included_memory_ids,
        "token_estimate": result.token_estimate,
        "candidates": [candidate_to_dict(c) for c in result.candidates],
        "gate_decisions": [gate_decision_to_dict(d) for d in result.gate_decisions],
        "metadata": result.metadata,
    }
    if include_context:
        row["context_text"] = result.context_text
    return row


def safe_json(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return str(obj)

