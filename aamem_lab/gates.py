from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .schemas import Candidate, ExperimentCase, GateDecision, MemoryRecord
from .text import contains_risk_terms, estimate_tokens, overlap_coeff, token_set


APPLY = "APPLY"
SUPPORT_ONLY = "SUPPORT_ONLY"
WARNING = "WARNING"
STALE = "STALE"
CONTRADICTED = "CONTRADICTED"
UNCERTAIN = "UNCERTAIN"
IRRELEVANT = "IRRELEVANT"


@dataclass
class GateConfig:
    apply_threshold: float = 0.58
    support_threshold: float = 0.36
    warning_threshold: float = 0.44
    contradiction_threshold: float = 0.74
    token_value_floor: float = 0.08
    semantic_weight: float = 0.45
    condition_weight: float = 0.25
    temporal_weight: float = 0.10
    utility_weight: float = 0.10
    risk_weight: float = 0.20
    token_cost_weight: float = 0.10


class HeuristicApplicabilityGate:
    """Offline gate for Phase 1.

    This gate intentionally uses only query, memory content, metadata, and
    candidate scores. It does not look at gold evidence ids. A later phase can
    replace it with an LLM judge or a trained classifier using the same output
    schema.
    """

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()

    def judge(
        self,
        case: ExperimentCase,
        candidates: Sequence[Candidate],
        evidence_pool: Sequence[MemoryRecord] | None = None,
    ) -> List[GateDecision]:
        if not candidates:
            return []
        max_score = max(abs(c.score) for c in candidates) or 1.0
        query_toks = token_set(case.query.query)
        recent_pool = list(evidence_pool or [])
        decisions: List[GateDecision] = []
        for candidate in candidates:
            memory = candidate.memory
            mem_toks = token_set(memory.content)
            semantic = max(0.0, min(1.0, candidate.score / max_score))
            condition = overlap_coeff(query_toks, mem_toks)
            temporal = self._temporal_validity(memory, recent_pool)
            utility = self._utility_score(memory)
            risk = self._contradiction_risk(case.query.query, memory, recent_pool)
            token_cost = min(1.0, estimate_tokens(memory.content) / 1200.0)
            token_value = max(0.0, semantic + condition - token_cost)
            applicability = (
                self.config.semantic_weight * semantic
                + self.config.condition_weight * condition
                + self.config.temporal_weight * temporal
                + self.config.utility_weight * utility
                - self.config.risk_weight * risk
                - self.config.token_cost_weight * token_cost
            )

            if risk >= self.config.contradiction_threshold and semantic >= self.config.warning_threshold:
                label = WARNING
                decision = "include_as_warning"
                reason = "Relevant memory has conflict/staleness risk terms; include only as a caution."
            elif applicability >= self.config.apply_threshold and token_value >= self.config.token_value_floor:
                label = APPLY
                decision = "use_as_premise"
                reason = "Relevant, condition-matching, and not high risk under current heuristic signals."
            elif applicability >= self.config.support_threshold:
                label = SUPPORT_ONLY
                decision = "include_as_background"
                reason = "Some relevance, but not strong enough to authorize as a premise."
            elif semantic >= self.config.warning_threshold and risk > 0.45:
                label = UNCERTAIN
                decision = "include_as_uncertain"
                reason = "Relevant but risk signal is too high for direct use."
            else:
                label = IRRELEVANT
                decision = "drop"
                reason = "Low estimated applicability for this query."

            decisions.append(
                GateDecision(
                    memory=memory,
                    label=label,
                    decision=decision,
                    scores={
                        "semantic_relevance": semantic,
                        "condition_match": condition,
                        "temporal_validity": temporal,
                        "utility": utility,
                        "contradiction_risk": risk,
                        "token_cost": token_cost,
                        "token_value": token_value,
                        "applicability": applicability,
                    },
                    reason=reason,
                    rank=candidate.rank,
                    source="heuristic_gate",
                    expanded_from=candidate.expanded_from,
                )
            )
        return decisions

    def _temporal_validity(self, memory: MemoryRecord, recent_pool: Sequence[MemoryRecord]) -> float:
        # Phase 1 has heterogeneous timestamp strings, so this is conservative.
        if not recent_pool or not memory.timestamp:
            return 0.7
        same_sample = [m for m in recent_pool if m.sample_id == memory.sample_id]
        if not same_sample:
            return 0.7
        return 0.8

    def _utility_score(self, memory: MemoryRecord) -> float:
        utility = memory.metadata.get("utility")
        if isinstance(utility, dict):
            used = float(utility.get("used_count", 0.0))
            success = float(utility.get("success_count", 0.0))
            failure = float(utility.get("failure_count", 0.0))
            if used > 0:
                return max(0.0, min(1.0, (success + 0.5) / (used + failure + 1.0)))
        return 0.5

    def _contradiction_risk(
        self,
        query: str,
        memory: MemoryRecord,
        recent_pool: Sequence[MemoryRecord],
    ) -> float:
        risk = 0.0
        if contains_risk_terms(query):
            risk += 0.15
        if contains_risk_terms(memory.content):
            risk += 0.35
        mtoks = token_set(memory.content)
        for other in recent_pool[:8]:
            if other.id == memory.id:
                continue
            if contains_risk_terms(other.content) and overlap_coeff(mtoks, token_set(other.content)) > 0.15:
                risk += 0.25
                break
        return min(1.0, risk)


class GoldOracleGate:
    """Diagnostic upper bound. This is not a deployable method."""

    def judge(
        self,
        case: ExperimentCase,
        candidates: Sequence[Candidate],
        evidence_pool: Sequence[MemoryRecord] | None = None,
    ) -> List[GateDecision]:
        gold = case.gold_set
        decisions: List[GateDecision] = []
        for candidate in candidates:
            is_gold = candidate.memory.id in gold
            label = APPLY if is_gold else IRRELEVANT
            decisions.append(
                GateDecision(
                    memory=candidate.memory,
                    label=label,
                    decision="use_as_premise" if is_gold else "drop",
                    scores={"oracle_gold": 1.0 if is_gold else 0.0},
                    reason="Gold evidence id match." if is_gold else "Not annotated as gold evidence.",
                    rank=candidate.rank,
                    source="gold_oracle_gate",
                    expanded_from=candidate.expanded_from,
                )
            )
        return decisions

