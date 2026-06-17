from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MemoryRecord:
    """A normalized memory unit used by all datasets and methods."""

    id: str
    content: str
    source_dataset: str
    sample_id: str
    timestamp: Optional[str] = None
    memory_type: str = "episodic"
    scope: str = "case"
    speaker: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    is_gold: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def short(self, max_chars: int = 240) -> str:
        text = " ".join(self.content.split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."


@dataclass(frozen=True)
class QueryExample:
    id: str
    query: str
    answer: str
    source_dataset: str
    sample_id: str
    gold_memory_ids: List[str] = field(default_factory=list)
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentCase:
    """One query plus the memory corpus available to answer it."""

    query: QueryExample
    memories: List[MemoryRecord]

    @property
    def gold_set(self) -> set[str]:
        return set(self.query.gold_memory_ids)


@dataclass(frozen=True)
class Candidate:
    memory: MemoryRecord
    score: float
    rank: int
    source: str = "retriever"
    expanded_from: Optional[str] = None


@dataclass(frozen=True)
class GateDecision:
    memory: MemoryRecord
    label: str
    decision: str
    scores: Dict[str, float]
    reason: str
    rank: int
    source: str = "gate"
    expanded_from: Optional[str] = None


@dataclass
class MethodResult:
    method: str
    case_id: str
    included_memory_ids: List[str]
    context_text: str
    token_estimate: int
    candidates: List[Candidate] = field(default_factory=list)
    gate_decisions: List[GateDecision] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

