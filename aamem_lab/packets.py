from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, List, Sequence

from .gates import APPLY, SUPPORT_ONLY, UNCERTAIN, WARNING
from .schemas import Candidate, GateDecision
from .text import estimate_tokens, truncate_to_token_budget


def render_raw_context(candidates: Sequence[Candidate], token_budget: int = 2000) -> str:
    lines: List[str] = ["RAW RETRIEVED MEMORY"]
    for cand in candidates:
        mem = cand.memory
        origin = f" expanded_from={cand.expanded_from}" if cand.expanded_from else ""
        lines.append(
            f"[{cand.rank}] id={mem.id} score={cand.score:.4f}{origin}\n"
            f"time={mem.timestamp or ''}\n"
            f"{mem.content}\n"
        )
    return truncate_to_token_budget("\n".join(lines), token_budget)


def render_authorized_packet(
    decisions: Sequence[GateDecision],
    token_budget: int = 1200,
    include_reasons: bool = True,
) -> str:
    sections = OrderedDict(
        [
            ("Applicable facts", []),
            ("Support-only background", []),
            ("Warnings / stale-premise guards", []),
            ("Uncertain memories", []),
        ]
    )
    for dec in decisions:
        if dec.label == APPLY:
            target = "Applicable facts"
        elif dec.label == SUPPORT_ONLY:
            target = "Support-only background"
        elif dec.label == WARNING:
            target = "Warnings / stale-premise guards"
        elif dec.label == UNCERTAIN:
            target = "Uncertain memories"
        else:
            continue
        score = dec.scores.get("applicability", dec.scores.get("oracle_gold", 0.0))
        reason = f"\n  reason: {dec.reason}" if include_reasons else ""
        sections[target].append(
            f"- [{dec.memory.id}] label={dec.label} score={score:.3f}\n"
            f"  {dec.memory.short(420)}{reason}"
        )

    lines = [
        "AUTHORIZED MEMORY PACKET",
        "Use applicable facts as current-state premises.",
        "Use support-only memories only as background.",
        "Do not use warning/stale memories as premises; correct the user if the query assumes them.",
        "",
    ]
    for title, items in sections.items():
        lines.append(f"{title}:")
        lines.extend(items if items else ["- None"])
        lines.append("")
    return truncate_to_token_budget("\n".join(lines), token_budget)


def select_decisions_for_packet(
    decisions: Sequence[GateDecision],
    token_budget: int = 1200,
    max_apply: int = 8,
    max_support: int = 3,
    max_warning: int = 3,
    include_uncertain: bool = False,
) -> List[GateDecision]:
    """Select the decisions that will actually enter the prompt packet.

    This is separate from judging. The gate can label many candidates, but the
    packet must be budget-aware; metrics should count only memories that are
    really made available to the answerer.
    """

    limits = {
        APPLY: max_apply,
        WARNING: max_warning,
        SUPPORT_ONLY: max_support,
        UNCERTAIN: 2 if include_uncertain else 0,
    }
    priority = {APPLY: 0, WARNING: 1, SUPPORT_ONLY: 2, UNCERTAIN: 3}
    selected: List[GateDecision] = []
    used_by_label = {label: 0 for label in limits}
    used_tokens = 90

    eligible = [d for d in decisions if d.label in limits and limits[d.label] > 0]
    eligible.sort(
        key=lambda d: (
            priority.get(d.label, 99),
            -float(d.scores.get("applicability", d.scores.get("oracle_gold", 0.0))),
            d.rank,
        )
    )

    for decision in eligible:
        if used_by_label[decision.label] >= limits[decision.label]:
            continue
        cost = estimate_tokens(decision.memory.short(420)) + 35
        if selected and used_tokens + cost > token_budget:
            continue
        selected.append(decision)
        used_by_label[decision.label] += 1
        used_tokens += cost
    return selected


def included_ids_from_decisions(decisions: Iterable[GateDecision]) -> List[str]:
    keep = {APPLY, SUPPORT_ONLY, WARNING, UNCERTAIN}
    ids: List[str] = []
    for dec in decisions:
        if dec.label in keep and dec.memory.id not in ids:
            ids.append(dec.memory.id)
    return ids


def token_estimate(text: str) -> int:
    return estimate_tokens(text)
