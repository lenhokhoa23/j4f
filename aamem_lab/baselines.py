from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Dict, List, Sequence

from .gates import APPLY, SUPPORT_ONLY, GoldOracleGate, HeuristicApplicabilityGate, STALE
from .packets import (
    included_ids_from_decisions,
    render_authorized_packet,
    render_raw_context,
    select_decisions_for_packet,
)
from .retrieval import BM25Retriever, build_token_knn_links
from .schemas import Candidate, ExperimentCase, GateDecision, MethodResult, MemoryRecord
from .text import estimate_tokens, overlap_coeff, token_set


class MemoryMethod(ABC):
    name: str

    @abstractmethod
    def run(self, case: ExperimentCase) -> MethodResult:
        raise NotImplementedError


def _dedupe_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    seen = set()
    out: List[Candidate] = []
    for cand in candidates:
        if cand.memory.id in seen:
            continue
        seen.add(cand.memory.id)
        out.append(cand)
    return [
        Candidate(
            memory=c.memory,
            score=c.score,
            rank=i + 1,
            source=c.source,
            expanded_from=c.expanded_from,
        )
        for i, c in enumerate(out)
    ]


class RawTopKMethod(MemoryMethod):
    def __init__(self, k: int = 5, token_budget: int = 2000):
        self.k = k
        self.token_budget = token_budget
        self.name = f"raw_topk_k{self.k}"

    def run(self, case: ExperimentCase) -> MethodResult:
        candidates = BM25Retriever(case.memories).search(case.query.query, self.k)
        context = render_raw_context(candidates, self.token_budget)
        return MethodResult(
            method=self.name,
            case_id=case.query.id,
            included_memory_ids=[c.memory.id for c in candidates],
            context_text=context,
            token_estimate=estimate_tokens(context),
            candidates=candidates,
        )


class AMemStyleBoxMethod(MemoryMethod):
    """A-MEM-style baseline: retrieve seeds, then pull linked neighborhood.

    This mirrors the behavior we observed in A-MEM's raw retrieval path:
    top-k seeds are retrieved, then linked memories/neighbors are appended into
    the prompt without an applicability check.
    """

    def __init__(
        self,
        seed_k: int = 5,
        neighbor_k: int = 5,
        token_budget: int = 4000,
        min_link_jaccard: float = 0.04,
    ):
        self.seed_k = seed_k
        self.neighbor_k = neighbor_k
        self.token_budget = token_budget
        self.min_link_jaccard = min_link_jaccard
        self.name = f"amem_style_box_seed{seed_k}_nbr{neighbor_k}"

    def run(self, case: ExperimentCase) -> MethodResult:
        retriever = BM25Retriever(case.memories)
        seeds = retriever.search(case.query.query, self.seed_k)
        by_id: Dict[str, MemoryRecord] = {m.id: m for m in case.memories}
        links = build_token_knn_links(
            case.memories,
            top_n=self.neighbor_k,
            min_jaccard=self.min_link_jaccard,
        )
        expanded: List[Candidate] = []
        for seed in seeds:
            expanded.append(seed)
            for nbr_id in links.get(seed.memory.id, [])[: self.neighbor_k]:
                nbr = by_id.get(nbr_id)
                if not nbr:
                    continue
                expanded.append(
                    Candidate(
                        memory=nbr,
                        score=max(0.0, seed.score * 0.75),
                        rank=len(expanded) + 1,
                        source="amem_style_neighbor",
                        expanded_from=seed.memory.id,
                    )
                )
        candidates = _dedupe_candidates(expanded)
        context = render_raw_context(candidates, self.token_budget)
        return MethodResult(
            method=self.name,
            case_id=case.query.id,
            included_memory_ids=[c.memory.id for c in candidates],
            context_text=context,
            token_estimate=estimate_tokens(context),
            candidates=candidates,
            metadata={"seed_count": len(seeds), "expanded_count": len(candidates)},
        )


class AAMemMethod(MemoryMethod):
    def __init__(
        self,
        candidate_k: int = 12,
        seed_k: int = 5,
        neighbor_k: int = 3,
        packet_budget: int = 1200,
    ):
        self.candidate_k = candidate_k
        self.seed_k = seed_k
        self.neighbor_k = neighbor_k
        self.packet_budget = packet_budget
        self.gate = HeuristicApplicabilityGate()
        self.name = f"aamem_candidate{candidate_k}_packet{packet_budget}"

    def run(self, case: ExperimentCase) -> MethodResult:
        retriever = BM25Retriever(case.memories)
        seeds = retriever.search(case.query.query, self.seed_k)
        pool = retriever.search(case.query.query, self.candidate_k)

        by_id: Dict[str, MemoryRecord] = {m.id: m for m in case.memories}
        links = build_token_knn_links(case.memories, top_n=self.neighbor_k)
        expanded = list(pool)
        for seed in seeds:
            for nbr_id in links.get(seed.memory.id, [])[: self.neighbor_k]:
                nbr = by_id.get(nbr_id)
                if nbr:
                    expanded.append(
                        Candidate(
                            memory=nbr,
                            score=max(0.0, seed.score * 0.60),
                            rank=len(expanded) + 1,
                            source="aamem_evidence_neighbor",
                            expanded_from=seed.memory.id,
                        )
                    )
        candidates = _dedupe_candidates(expanded)
        evidence_pool = [c.memory for c in candidates]
        decisions = self.gate.judge(case, candidates, evidence_pool=evidence_pool)
        packet_decisions = select_decisions_for_packet(
            decisions,
            token_budget=self.packet_budget,
            max_apply=max(2, self.seed_k),
            max_support=2,
            max_warning=2,
            include_uncertain=False,
        )
        packet = render_authorized_packet(packet_decisions, self.packet_budget)
        return MethodResult(
            method=self.name,
            case_id=case.query.id,
            included_memory_ids=included_ids_from_decisions(packet_decisions),
            context_text=packet,
            token_estimate=estimate_tokens(packet),
            candidates=candidates,
            gate_decisions=decisions,
            metadata={
                "seed_count": len(seeds),
                "candidate_count": len(candidates),
                "packet_decision_count": len(packet_decisions),
            },
        )


UPDATE_TERMS = {
    "changed",
    "updated",
    "instead",
    "replaced",
    "moved",
    "switched",
    "not",
    "stopped",
    "cancelled",
}


def _timestamp_key(value: str | None) -> tuple:
    if not value:
        return (0, "")
    nums = tuple(int(x) for x in re.findall(r"\d+", value)[:6])
    return (1, nums, value)


class AAMemStaleGuardMethod(AAMemMethod):
    """AAMem variant with an explicit stale-memory suppression pass.

    It does not use gold stale labels. It looks for a newer candidate with
    update/change language and high lexical overlap with an older candidate.
    The older candidate is marked STALE and excluded from the authorized packet.
    """

    def __init__(
        self,
        candidate_k: int = 14,
        seed_k: int = 5,
        neighbor_k: int = 3,
        packet_budget: int = 1200,
        stale_overlap_threshold: float = 0.18,
    ):
        super().__init__(
            candidate_k=candidate_k,
            seed_k=seed_k,
            neighbor_k=neighbor_k,
            packet_budget=packet_budget,
        )
        self.stale_overlap_threshold = stale_overlap_threshold
        self.name = f"aamem_stale_guard_candidate{candidate_k}_packet{packet_budget}"

    def _apply_stale_guard(self, decisions: Sequence[GateDecision]) -> List[GateDecision]:
        stale_pairs: Dict[str, GateDecision] = {}
        for decision in decisions:
            memory = decision.memory
            mem_tokens = token_set(memory.content)
            for other in decisions:
                if other.memory.id == memory.id:
                    continue
                if _timestamp_key(other.memory.timestamp) <= _timestamp_key(memory.timestamp):
                    continue
                other_tokens = token_set(other.memory.content)
                if not (other_tokens & UPDATE_TERMS):
                    continue
                if overlap_coeff(mem_tokens, other_tokens) < self.stale_overlap_threshold:
                    continue
                stale_pairs[memory.id] = other
                break

        promote_ids = {newer.memory.id for newer in stale_pairs.values()}
        updated: List[GateDecision] = []
        for decision in decisions:
            stale_evidence = stale_pairs.get(decision.memory.id)
            if stale_evidence and decision.label in {APPLY, SUPPORT_ONLY}:
                scores = dict(decision.scores)
                scores["stale_guard_overlap"] = overlap_coeff(
                    token_set(decision.memory.content),
                    token_set(stale_evidence.memory.content),
                )
                updated.append(
                    GateDecision(
                        memory=decision.memory,
                        label=STALE,
                        decision="drop_stale_candidate",
                        scores=scores,
                        reason=(
                            "A newer overlapping memory contains update/change language; "
                            f"newer_id={stale_evidence.memory.id}."
                        ),
                        rank=decision.rank,
                        source="stale_guard",
                        expanded_from=decision.expanded_from,
                    )
                )
            elif decision.memory.id in promote_ids and decision.label != APPLY:
                scores = dict(decision.scores)
                scores["applicability"] = max(float(scores.get("applicability", 0.0)), 0.82)
                scores["stale_guard_promoted_current_update"] = 1.0
                updated.append(
                    GateDecision(
                        memory=decision.memory,
                        label=APPLY,
                        decision="use_as_current_update",
                        scores=scores,
                        reason="Promoted because this newer update invalidates an older overlapping memory.",
                        rank=decision.rank,
                        source="stale_guard",
                        expanded_from=decision.expanded_from,
                    )
                )
            else:
                updated.append(decision)
        return updated

    def run(self, case: ExperimentCase) -> MethodResult:
        result = super().run(case)
        guarded_decisions = self._apply_stale_guard(result.gate_decisions)
        packet_decisions = select_decisions_for_packet(
            guarded_decisions,
            token_budget=self.packet_budget,
            max_apply=max(2, self.seed_k),
            max_support=2,
            max_warning=2,
            include_uncertain=False,
        )
        packet = render_authorized_packet(packet_decisions, self.packet_budget)
        return MethodResult(
            method=self.name,
            case_id=case.query.id,
            included_memory_ids=included_ids_from_decisions(packet_decisions),
            context_text=packet,
            token_estimate=estimate_tokens(packet),
            candidates=result.candidates,
            gate_decisions=guarded_decisions,
            metadata={
                **result.metadata,
                "stale_guard": True,
                "stale_decision_count": sum(1 for d in guarded_decisions if d.label == STALE),
                "packet_decision_count": len(packet_decisions),
            },
        )


class OracleApplicabilityMethod(MemoryMethod):
    """Upper bound for retrieval+packing if applicability labels were perfect."""

    def __init__(self, candidate_k: int = 30, packet_budget: int = 1200):
        self.candidate_k = candidate_k
        self.packet_budget = packet_budget
        self.gate = GoldOracleGate()
        self.name = f"oracle_applicability_candidate{candidate_k}"

    def run(self, case: ExperimentCase) -> MethodResult:
        candidates = BM25Retriever(case.memories).search(case.query.query, self.candidate_k)
        decisions = self.gate.judge(case, candidates)
        packet_decisions = select_decisions_for_packet(
            decisions,
            token_budget=self.packet_budget,
            max_apply=self.candidate_k,
            max_support=0,
            max_warning=0,
            include_uncertain=False,
        )
        packet = render_authorized_packet(packet_decisions, self.packet_budget, include_reasons=False)
        return MethodResult(
            method=self.name,
            case_id=case.query.id,
            included_memory_ids=included_ids_from_decisions(packet_decisions),
            context_text=packet,
            token_estimate=estimate_tokens(packet),
            candidates=candidates,
            gate_decisions=decisions,
            metadata={"oracle": True, "packet_decision_count": len(packet_decisions)},
        )


def build_methods(names: Sequence[str], k: int = 5) -> List[MemoryMethod]:
    methods: List[MemoryMethod] = []
    for name in names:
        if name == "raw_topk":
            methods.append(RawTopKMethod(k=k))
        elif name == "amem_box":
            methods.append(AMemStyleBoxMethod(seed_k=k, neighbor_k=k))
        elif name == "aamem":
            methods.append(AAMemMethod(seed_k=k, candidate_k=max(k * 2, 10)))
        elif name == "aamem_stale_guard":
            methods.append(AAMemStaleGuardMethod(seed_k=k, candidate_k=max(k * 3, 14)))
        elif name == "oracle":
            methods.append(OracleApplicabilityMethod(candidate_k=max(k * 4, 20)))
        else:
            raise ValueError(f"Unknown method: {name}")
    return methods
