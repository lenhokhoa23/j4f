from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SotaReference:
    key: str
    name: str
    year: int
    role_in_this_lab: str
    comparison_status: str
    notes: str
    source_url: str = ""


SOTA_REFERENCES: Dict[str, SotaReference] = {
    "amem": SotaReference(
        key="amem",
        name="A-MEM / Agentic Memory",
        year=2025,
        role_in_this_lab="Main structural baseline: retrieve top-k, then expand linked/neighborhood memories.",
        comparison_status="Implemented as amem_box proxy; optional official wrapper is provided.",
        notes=(
            "The proxy focuses on the retrieval-time behavior that matters for our bottleneck: "
            "neighbors are appended to the prompt without an applicability verifier."
        ),
        source_url="",
    ),
    "actmem": SotaReference(
        key="actmem",
        name="ActMem / ActMemEval",
        year=2026,
        role_in_this_lab="Primary evaluation dataset for action-conditioned memory retrieval.",
        comparison_status="Dataset loader implemented; method comparison runs locally.",
        notes=(
            "ActMemEval gives answer_session_ids, so it is useful for measuring whether a method "
            "keeps the correct evidence while reducing irrelevant memory."
        ),
        source_url="https://arxiv.org/html/2603.00026v1",
    ),
    "stale": SotaReference(
        key="stale",
        name="STALE / invalidated-memory evaluation",
        year=2026,
        role_in_this_lab="Primary 2026 target for stale/obsolete memory handling.",
        comparison_status="Implemented as synthetic_stale Phase 3 plus aamem_stale_guard; official STALE loader remains future work.",
        notes=(
            "STALE targets invalidated memories and implicit conflict. Phase 3 measures fresh-memory hit, "
            "stale premise leak, stale answer leak, and guard rate."
        ),
        source_url="https://arxiv.org/abs/2605.06527",
    ),
    "deltamem": SotaReference(
        key="deltamem",
        name="DeltaMem-style memory update/compression",
        year=2026,
        role_in_this_lab="Later phase for memory consolidation, not first-pass retrieval.",
        comparison_status="Not implemented yet.",
        notes=(
            "Delta-style methods attack memory growth/update. Our current phase attacks whether a "
            "retrieved memory is authorized for the current answer."
        ),
        source_url="https://arxiv.org/html/2606.03083v1",
    ),
    "memoryarena": SotaReference(
        key="memoryarena",
        name="MemoryArena",
        year=2026,
        role_in_this_lab="Future benchmark target for multi-session memory-action loops.",
        comparison_status="Not implemented in this lightweight runner yet.",
        notes=(
            "MemoryArena evaluates memory inside interdependent multi-session tasks, so it is stronger "
            "than static QA once our gate is stable."
        ),
        source_url="https://arxiv.org/abs/2602.16313",
    ),
    "memoryagentbench": SotaReference(
        key="memoryagentbench",
        name="MemoryAgentBench",
        year=2026,
        role_in_this_lab="Future benchmark target for incremental multi-turn memory interactions.",
        comparison_status="Not implemented in this lightweight runner yet.",
        notes="Useful after Phase 2 answerer is stable because it tests memory over incremental interactions.",
        source_url="https://github.com/HUST-AI-HYZ/MemoryAgentBench",
    ),
}


def list_references() -> List[SotaReference]:
    return list(SOTA_REFERENCES.values())


def references_as_dict() -> List[dict]:
    return [ref.__dict__ for ref in list_references()]
