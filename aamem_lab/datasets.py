from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import ExperimentCase, MemoryRecord, QueryExample
from .text import extract_evidence_ids, normalize_text


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _dialogue_to_text(messages: Iterable[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role") or msg.get("speaker") or "speaker"
        content = msg.get("content") or msg.get("text") or ""
        if content:
            parts.append(f"{role}: {normalize_text(content)}")
        if msg.get("blip_caption"):
            parts.append(f"{role} image: {normalize_text(str(msg['blip_caption']))}")
        if msg.get("query"):
            parts.append(f"{role} image-query: {normalize_text(str(msg['query']))}")
    return "\n".join(parts)


def load_actmem_eval(
    path: str | Path,
    limit: Optional[int] = None,
    max_sessions_per_case: Optional[int] = None,
) -> List[ExperimentCase]:
    """Load ActMemEval as one case per benchmark question.

    The ActMem release already marks answer_session_ids. We use those ids only
    for evaluation labels, never inside methods.
    """

    data = _load_json(path)
    cases: List[ExperimentCase] = []
    for idx, item in enumerate(data[:limit] if limit else data):
        sample_id = item.get("question_id") or f"actmem_{idx}"
        gold_ids = [str(x) for x in item.get("answer_session_ids", [])]
        sessions = item.get("haystack_sessions", [])
        session_ids = item.get("haystack_session_ids", [])
        dates = item.get("haystack_dates", [])
        if max_sessions_per_case:
            sessions = sessions[:max_sessions_per_case]
            session_ids = session_ids[:max_sessions_per_case]
            dates = dates[:max_sessions_per_case]

        memories: List[MemoryRecord] = []
        for sidx, session in enumerate(sessions):
            mid = str(session_ids[sidx]) if sidx < len(session_ids) else f"{sample_id}:session_{sidx}"
            timestamp = str(dates[sidx]) if sidx < len(dates) else None
            content = _dialogue_to_text(session)
            memories.append(
                MemoryRecord(
                    id=mid,
                    content=content,
                    source_dataset="actmem_eval",
                    sample_id=sample_id,
                    timestamp=timestamp,
                    memory_type="dialogue_session",
                    is_gold=mid in gold_ids,
                    metadata={"session_index": sidx},
                )
            )

        query = QueryExample(
            id=str(sample_id),
            query=normalize_text(item.get("question", "")),
            answer=normalize_text(item.get("answer", "")),
            source_dataset="actmem_eval",
            sample_id=sample_id,
            gold_memory_ids=gold_ids,
            metadata={"question_date": item.get("question_date")},
        )
        cases.append(ExperimentCase(query=query, memories=memories))
    return cases


SESSION_KEY_RE = re.compile(r"session_(\d+)$")


def _iter_locomo_turns(conversation: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key, value in conversation.items():
        match = SESSION_KEY_RE.fullmatch(key)
        if not match or not isinstance(value, list):
            continue
        session_num = int(match.group(1))
        date_key = f"session_{session_num}_date_time"
        timestamp = conversation.get(date_key)
        for turn in value:
            if not isinstance(turn, dict):
                continue
            yield {
                "session_num": session_num,
                "timestamp": timestamp,
                "speaker": turn.get("speaker"),
                "dia_id": turn.get("dia_id"),
                "text": turn.get("text"),
                "blip_caption": turn.get("blip_caption"),
                "query": turn.get("query"),
            }


def _locomo_turn_content(turn: Dict[str, Any]) -> str:
    parts = []
    speaker = turn.get("speaker") or "speaker"
    text = turn.get("text")
    if text:
        parts.append(f"{speaker}: {normalize_text(str(text))}")
    if turn.get("blip_caption"):
        parts.append(f"{speaker} image: {normalize_text(str(turn['blip_caption']))}")
    if turn.get("query"):
        parts.append(f"{speaker} image-query: {normalize_text(str(turn['query']))}")
    return "\n".join(parts)


def load_locomo(
    path: str | Path,
    limit_samples: Optional[int] = None,
    limit_questions_per_sample: Optional[int] = None,
) -> List[ExperimentCase]:
    """Load LoCoMo local subset as turn-level memories.

    Gold labels are mapped from evidence ids like D1:3 to memory ids
    ``{sample_id}:D1:3``. Empty evidence remains empty, preserving
    unanswerable/adversarial cases.
    """

    data = _load_json(path)
    selected = data[:limit_samples] if limit_samples else data
    cases: List[ExperimentCase] = []

    for sidx, sample in enumerate(selected):
        sample_id = str(sample.get("sample_id") or f"locomo_{sidx}")
        conversation = sample.get("conversation") or {}
        memories: List[MemoryRecord] = []
        for turn in _iter_locomo_turns(conversation):
            dia_id = turn.get("dia_id")
            if not dia_id:
                continue
            mid = f"{sample_id}:{dia_id}"
            memories.append(
                MemoryRecord(
                    id=mid,
                    content=_locomo_turn_content(turn),
                    source_dataset="locomo",
                    sample_id=sample_id,
                    timestamp=turn.get("timestamp"),
                    memory_type="dialogue_turn",
                    speaker=turn.get("speaker"),
                    metadata={
                        "dia_id": dia_id,
                        "session_num": turn.get("session_num"),
                    },
                )
            )

        qas = sample.get("qa") or []
        if limit_questions_per_sample:
            qas = qas[:limit_questions_per_sample]
        for qidx, qa in enumerate(qas):
            evidence_ids = extract_evidence_ids(qa.get("evidence", []))
            gold_ids = [f"{sample_id}:{eid}" for eid in evidence_ids]
            query = QueryExample(
                id=f"{sample_id}:q{qidx}",
                query=normalize_text(str(qa.get("question", ""))),
                answer=normalize_text(str(qa.get("answer", ""))),
                source_dataset="locomo",
                sample_id=sample_id,
                gold_memory_ids=gold_ids,
                category=str(qa.get("category")) if qa.get("category") is not None else None,
                metadata={"raw_evidence": qa.get("evidence", [])},
            )
            cases.append(ExperimentCase(query=query, memories=memories))
    return cases


def load_cases(
    dataset: str,
    path: str | Path,
    limit: Optional[int] = None,
    limit_samples: Optional[int] = None,
    limit_questions_per_sample: Optional[int] = None,
) -> List[ExperimentCase]:
    if dataset == "actmem":
        return load_actmem_eval(path, limit=limit)
    if dataset == "locomo":
        return load_locomo(
            path,
            limit_samples=limit_samples if limit_samples is not None else limit,
            limit_questions_per_sample=limit_questions_per_sample,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")

