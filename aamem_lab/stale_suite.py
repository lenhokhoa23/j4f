from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from .answer_metrics import normalize_answer, score_answer
from .gates import APPLY, SUPPORT_ONLY, WARNING, STALE, CONTRADICTED, UNCERTAIN
from .schemas import ExperimentCase, GateDecision, MemoryRecord, MethodResult, QueryExample


@dataclass(frozen=True)
class StaleScenario:
    key: str
    question: str
    fresh_answer: str
    stale_answer: str
    old_memory: str
    new_memory: str
    distractors: Sequence[str]


SCENARIOS: List[StaleScenario] = [
    StaleScenario(
        key="office_location",
        question="Where is the user's current office?",
        fresh_answer="Seattle",
        stale_answer="Boston",
        old_memory="2024-03-10: The user's office is in Boston near South Station.",
        new_memory="2026-02-18: The user moved their current office to Seattle instead of Boston.",
        distractors=[
            "2025-06-02: The user visited Boston for a client meeting.",
            "2026-02-20: The Seattle office has a parking garage and a quiet room.",
        ],
    ),
    StaleScenario(
        key="diet_preference",
        question="What is the user's current dinner preference?",
        fresh_answer="pescatarian",
        stale_answer="vegan",
        old_memory="2024-11-01: The user follows a vegan dinner preference.",
        new_memory="2026-01-15: The user changed their current dinner preference to pescatarian, not vegan.",
        distractors=[
            "2025-09-09: The user liked a vegan bakery during a trip.",
            "2026-01-20: The user asked for seafood options that avoid dairy.",
        ],
    ),
    StaleScenario(
        key="meeting_time",
        question="When is the current team sync?",
        fresh_answer="Wednesday at 2 PM",
        stale_answer="Monday at 9 AM",
        old_memory="2024-05-07: The team sync is Monday at 9 AM.",
        new_memory="2026-03-22: The team sync changed to Wednesday at 2 PM instead of Monday at 9 AM.",
        distractors=[
            "2025-12-01: Monday at 9 AM was used for a one-off planning call.",
            "2026-03-23: Wednesday at 2 PM is now reserved on the calendar.",
        ],
    ),
    StaleScenario(
        key="project_deadline",
        question="What is the current deadline for Project Orion?",
        fresh_answer="August 30",
        stale_answer="July 15",
        old_memory="2025-04-04: Project Orion deadline is July 15.",
        new_memory="2026-04-12: Project Orion deadline was updated to August 30, replacing July 15.",
        distractors=[
            "2025-07-16: The July 15 deadline caused a status review.",
            "2026-04-13: August 30 is written in the latest project tracker.",
        ],
    ),
    StaleScenario(
        key="preferred_language",
        question="Which programming language should be used for the current prototype?",
        fresh_answer="Python",
        stale_answer="JavaScript",
        old_memory="2025-01-08: The prototype should be written in JavaScript.",
        new_memory="2026-05-03: The current prototype switched to Python instead of JavaScript.",
        distractors=[
            "2025-01-09: JavaScript was useful for an earlier UI mockup.",
            "2026-05-04: Python was chosen because the evaluation code is notebook-first.",
        ],
    ),
    StaleScenario(
        key="shipping_address",
        question="What is the user's current shipping city?",
        fresh_answer="Austin",
        stale_answer="Denver",
        old_memory="2024-08-21: The user's shipping city is Denver.",
        new_memory="2026-01-30: The user moved and their current shipping city is Austin, not Denver.",
        distractors=[
            "2025-02-12: A package was delayed in Denver.",
            "2026-02-01: Austin is the city to use for current deliveries.",
        ],
    ),
    StaleScenario(
        key="account_plan",
        question="What account plan is currently active?",
        fresh_answer="Team plan",
        stale_answer="Free plan",
        old_memory="2025-03-05: The account is on the Free plan.",
        new_memory="2026-06-01: The account plan changed to the Team plan instead of the Free plan.",
        distractors=[
            "2025-05-10: The Free plan had a monthly export limit.",
            "2026-06-02: The Team plan includes shared workspaces.",
        ],
    ),
    StaleScenario(
        key="preferred_airport",
        question="Which airport should be used as the current default departure airport?",
        fresh_answer="SFO",
        stale_answer="JFK",
        old_memory="2024-12-12: The user's default departure airport is JFK.",
        new_memory="2026-04-18: The default departure airport is now SFO instead of JFK.",
        distractors=[
            "2025-03-03: JFK was used for a New York trip.",
            "2026-04-19: SFO is closest to the user's current home.",
        ],
    ),
    StaleScenario(
        key="newsletter_topic",
        question="What is the current topic of the monthly newsletter?",
        fresh_answer="memory agents",
        stale_answer="cloud security",
        old_memory="2025-09-30: The monthly newsletter topic is cloud security.",
        new_memory="2026-05-15: The monthly newsletter topic changed to memory agents instead of cloud security.",
        distractors=[
            "2025-10-01: Cloud security remained a sidebar topic.",
            "2026-05-16: The memory agents issue needs examples about stale memory.",
        ],
    ),
    StaleScenario(
        key="invoice_contact",
        question="Who is the current invoice contact?",
        fresh_answer="Mina",
        stale_answer="Jon",
        old_memory="2024-10-10: Jon is the invoice contact.",
        new_memory="2026-02-05: Mina replaced Jon as the current invoice contact.",
        distractors=[
            "2025-01-01: Jon approved an old invoice.",
            "2026-02-06: Mina should receive invoice questions now.",
        ],
    ),
    StaleScenario(
        key="cloud_region",
        question="Which cloud region is current for deployment?",
        fresh_answer="us-west-2",
        stale_answer="us-east-1",
        old_memory="2025-02-14: Deployment uses us-east-1.",
        new_memory="2026-03-28: Deployment moved to us-west-2 instead of us-east-1.",
        distractors=[
            "2025-03-01: us-east-1 was tested for latency.",
            "2026-03-29: us-west-2 is the latest deployment region.",
        ],
    ),
    StaleScenario(
        key="review_cadence",
        question="What is the current review cadence?",
        fresh_answer="biweekly",
        stale_answer="weekly",
        old_memory="2025-06-11: The review cadence is weekly.",
        new_memory="2026-05-20: The review cadence changed to biweekly, not weekly.",
        distractors=[
            "2025-06-12: Weekly reviews overloaded the team.",
            "2026-05-21: Biweekly review slots are now on the calendar.",
        ],
    ),
]


def _dimension_query_and_answer(scenario: StaleScenario, dimension: str) -> tuple[str, str]:
    if dimension == "SR":
        return (
            f"Is the old memory still valid: {scenario.stale_answer}?",
            f"No. The current value is {scenario.fresh_answer}, not {scenario.stale_answer}.",
        )
    if dimension == "PR":
        return (
            f"Since the stale value is {scenario.stale_answer}, answer the user's request: {scenario.question}",
            f"The premise is outdated. Use {scenario.fresh_answer} as the current value.",
        )
    if dimension == "IPA":
        return (scenario.question, scenario.fresh_answer)
    raise ValueError(f"Unknown stale probe dimension: {dimension}")


def build_synthetic_stale_cases(
    limit: int | None = None,
    noise_per_case: int = 4,
    seed: int = 7,
    dimensions: Sequence[str] = ("SR", "PR", "IPA"),
) -> List[ExperimentCase]:
    rng = random.Random(seed)
    selected = SCENARIOS[: limit or len(SCENARIOS)]
    cases: List[ExperimentCase] = []
    for idx, scenario in enumerate(selected):
        base_sample_id = f"synthetic_stale_{idx:03d}_{scenario.key}"
        old_id = f"{base_sample_id}:old"
        new_id = f"{base_sample_id}:new"
        base_memories = [
            MemoryRecord(
                id=old_id,
                content=scenario.old_memory,
                source_dataset="synthetic_stale",
                sample_id=base_sample_id,
                timestamp=scenario.old_memory.split(":", 1)[0],
                memory_type="episodic_update",
                metadata={"stale_role": "stale", "stale_answer": scenario.stale_answer},
            ),
            MemoryRecord(
                id=new_id,
                content=scenario.new_memory,
                source_dataset="synthetic_stale",
                sample_id=base_sample_id,
                timestamp=scenario.new_memory.split(":", 1)[0],
                memory_type="episodic_update",
                is_gold=True,
                metadata={"stale_role": "fresh", "fresh_answer": scenario.fresh_answer},
            ),
        ]
        distractors = list(scenario.distractors)
        while len(distractors) < noise_per_case:
            distractors.append(
                f"2026-01-{10 + len(distractors):02d}: Background note mentioning {scenario.stale_answer} "
                f"and {scenario.fresh_answer}, but not answering the current question."
            )
        rng.shuffle(distractors)
        for didx, text in enumerate(distractors[:noise_per_case]):
            base_memories.append(
                MemoryRecord(
                    id=f"{base_sample_id}:d{didx}",
                    content=text,
                    source_dataset="synthetic_stale",
                    sample_id=base_sample_id,
                    timestamp=text.split(":", 1)[0],
                    memory_type="distractor",
                    metadata={"stale_role": "distractor"},
                )
            )
        for dimension in dimensions:
            memories = list(base_memories)
            rng.shuffle(memories)
            question, answer = _dimension_query_and_answer(scenario, dimension)
            query = QueryExample(
                id=f"{base_sample_id}:{dimension}",
                query=question,
                answer=answer,
                source_dataset="synthetic_stale",
                sample_id=base_sample_id,
                gold_memory_ids=[new_id],
                category="stale_update",
                metadata={
                    "fresh_memory_ids": [new_id],
                    "stale_memory_ids": [old_id],
                    "fresh_answer": scenario.fresh_answer,
                    "stale_answer": scenario.stale_answer,
                    "probe_dimension": dimension,
                    "paper_metric_family": "STALE_SR_PR_IPA",
                },
            )
            cases.append(ExperimentCase(query=query, memories=memories))
    return cases


def _gate_label_map(decisions: Iterable[GateDecision]) -> Dict[str, str]:
    return {d.memory.id: d.label for d in decisions}


def score_stale_result(case: ExperimentCase, result: MethodResult, answer_text: str = "") -> Dict[str, float]:
    fresh_ids = set(case.query.metadata.get("fresh_memory_ids", case.query.gold_memory_ids))
    stale_ids = set(case.query.metadata.get("stale_memory_ids", []))
    included = set(result.included_memory_ids)
    labels = _gate_label_map(result.gate_decisions)
    premise_labels = {APPLY, SUPPORT_ONLY}
    guard_labels = {WARNING, STALE, CONTRADICTED, UNCERTAIN}

    if result.gate_decisions:
        stale_premise = {mid for mid in stale_ids if mid in included and labels.get(mid) in premise_labels}
        stale_guard = {mid for mid in stale_ids if labels.get(mid) in guard_labels}
    else:
        stale_premise = stale_ids & included
        stale_guard = set()

    fresh_hit = len(fresh_ids & included) / max(1, len(fresh_ids))
    stale_context_leak = len(stale_ids & included) / max(1, len(stale_ids))
    stale_premise_leak = len(stale_premise) / max(1, len(stale_ids))
    stale_guard_rate = len(stale_guard) / max(1, len(stale_ids))

    fresh_answer = str(case.query.metadata.get("fresh_answer", case.query.answer))
    stale_answer = str(case.query.metadata.get("stale_answer", ""))
    npred = normalize_answer(answer_text)
    nfresh = normalize_answer(fresh_answer)
    nstale = normalize_answer(stale_answer)
    fresh_answer_hit = 1.0 if nfresh and nfresh in npred else 0.0
    negated_stale_patterns = [
        f"instead of {nstale}",
        f"not {nstale}",
        f"rather than {nstale}",
        f"replacing {nstale}",
        f"replaced {nstale}",
        f"no longer {nstale}",
    ]
    stale_is_negated = bool(nstale and any(pattern in npred for pattern in negated_stale_patterns))
    stale_answer_hit = 1.0 if nstale and nstale in npred and not stale_is_negated else 0.0
    fresh_over_stale = 1.0 if fresh_answer_hit and not stale_answer_hit else 0.0
    ans = score_answer(fresh_answer, answer_text)
    dimension = str(case.query.metadata.get("probe_dimension", "IPA"))
    if dimension == "SR":
        stale_probe_accuracy = 1.0 if stale_guard_rate > 0 or (fresh_answer_hit and not stale_answer_hit) else 0.0
    elif dimension == "PR":
        stale_probe_accuracy = 1.0 if fresh_answer_hit and not stale_answer_hit and stale_premise_leak == 0 else 0.0
    else:
        stale_probe_accuracy = 1.0 if fresh_answer_hit and not stale_answer_hit else 0.0

    return {
        "probe_dimension": dimension,
        "fresh_memory_hit": fresh_hit,
        "stale_context_leak": stale_context_leak,
        "stale_premise_leak": stale_premise_leak,
        "stale_guard_rate": stale_guard_rate,
        "fresh_answer_hit": fresh_answer_hit,
        "stale_answer_leak": stale_answer_hit,
        "fresh_over_stale_answer": fresh_over_stale,
        "stale_probe_accuracy": stale_probe_accuracy,
        "answer_token_f1": ans.token_f1,
        "answer_rouge_l_f1": ans.rouge_l_f1,
    }


def aggregate_stale_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(rows)
    if not items:
        return {}
    keys = list(items[0].keys())
    out = {"n": float(len(items))}
    for key in keys:
        if key == "probe_dimension":
            continue
        out[key] = mean(float(row.get(key, 0.0)) for row in items)
    return out
