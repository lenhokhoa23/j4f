from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .text import estimate_tokens, overlap_coeff, token_set, tokenize


APPLY = "APPLY"
SUPPORT = "SUPPORT"
WARNING = "WARNING"
STALE = "STALE"
CONTRADICTED = "CONTRADICTED"
UNCERTAIN = "UNCERTAIN"
IRRELEVANT = "IRRELEVANT"

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
    "canceled",
    "deprecated",
    "no",
    "longer",
}

FAILURE_TERMS = {
    "failed",
    "failure",
    "crash",
    "crashed",
    "error",
    "blocked",
    "unsafe",
    "avoid",
    "risk",
    "wrong",
    "bug",
    "issue",
}


_HF_LLM_CACHE: Dict[str, Any] = {}
_OFFICIAL_METRIC_RESOURCES_READY = False


def _extract_backend_model(args: Sequence[Any], kwargs: Dict[str, Any]) -> tuple[str, str]:
    backend = kwargs.get("backend")
    model = kwargs.get("model")
    if backend is None and args:
        backend = args[0]
    if model is None and len(args) > 1:
        model = args[1]
    return str(backend or "hf"), str(model or "Qwen/Qwen2.5-1.5B-Instruct")


class HFLocalTextLLM:
    """Small HuggingFace local LLM with the same get_completion surface as A-MEM controllers."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."

    def __init__(self, model_name: str, max_new_tokens: int = 768):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HF backend requires transformers, torch, and accelerate. "
                "Install with: pip install transformers accelerate"
            ) from exc

        self.torch = torch
        print(f"[hf] loading model={model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()
        print("[hf] model loaded", flush=True)

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{self.SYSTEM_MESSAGE}\n\nUser:\n{prompt}\n\nAssistant:"

    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        text = self._format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt")
        try:
            device = self.model.device
        except AttributeError:
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
        inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = temperature is not None and temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(float(temperature), 1e-5)
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


class HFCompatibleRobustLLMController:
    """Drop-in replacement for official RobustLLMController when backend='hf'."""

    def __init__(self, *args: Any, original_cls: Optional[Any] = None, max_new_tokens: int = 768, **kwargs: Any):
        backend, model = _extract_backend_model(args, kwargs)
        if backend == "hf":
            cache_key = f"{model}::{max_new_tokens}"
            if cache_key not in _HF_LLM_CACHE:
                _HF_LLM_CACHE[cache_key] = HFLocalTextLLM(model, max_new_tokens=max_new_tokens)
            self.llm = _HF_LLM_CACHE[cache_key]
            self._delegate = None
        else:
            if original_cls is None:
                raise ValueError("original_cls is required for non-HF backends.")
            self._delegate = original_cls(*args, **kwargs)
            self.llm = self._delegate.llm


@dataclass
class AMemCandidate:
    local_id: str
    memory_index: int
    source: str
    seed_index: Optional[int]
    order: int
    timestamp: str
    content: str
    context: str
    keywords: List[str]
    tags: List[str]
    links: List[int] = field(default_factory=list)

    def render_official_block(self) -> str:
        """Match official RobustAgenticMemorySystem.find_related_memories_raw formatting."""

        return (
            "talk start time:" + self.timestamp
            + "memory content: " + self.content
            + "memory context: " + self.context
            + "memory keywords: " + str(self.keywords)
            + "memory tags: " + str(self.tags)
            + "\n"
        )


@dataclass
class GateDecision:
    local_id: str
    memory_index: int
    label: str
    usable_as_premise: bool
    confidence: float
    reason: str
    scores: Dict[str, float] = field(default_factory=dict)


class AMemApplicabilityGate:
    name = "base"

    def judge(
        self,
        question: str,
        retrieval_query: str,
        candidates: Sequence[AMemCandidate],
    ) -> List[GateDecision]:
        raise NotImplementedError


def _timestamp_key(value: str | None) -> tuple:
    if not value:
        return (0, "")
    nums = tuple(int(x) for x in re.findall(r"\d+", str(value))[:6])
    return (1, nums, str(value))


def _candidate_text(candidate: AMemCandidate) -> str:
    return " ".join(
        [
            candidate.content,
            candidate.context,
            " ".join(candidate.keywords),
            " ".join(candidate.tags),
        ]
    )


class HeuristicAMemApplicabilityGate(AMemApplicabilityGate):
    """Offline read-time A-MEM gate.

    This deliberately uses no gold labels. It is a transparent baseline for
    debugging the wrapper before replacing the gate with an LLM or trained
    verifier.
    """

    name = "heuristic"

    def __init__(
        self,
        apply_threshold: float = 0.46,
        support_threshold: float = 0.22,
        stale_overlap_threshold: float = 0.18,
    ):
        self.apply_threshold = apply_threshold
        self.support_threshold = support_threshold
        self.stale_overlap_threshold = stale_overlap_threshold

    def judge(
        self,
        question: str,
        retrieval_query: str,
        candidates: Sequence[AMemCandidate],
    ) -> List[GateDecision]:
        stale_by_newer = self._detect_stale_candidates(candidates)
        q_tokens = token_set(question + " " + retrieval_query)
        decisions: List[GateDecision] = []

        for cand in candidates:
            text = _candidate_text(cand)
            toks = token_set(text)
            relevance = overlap_coeff(q_tokens, toks)
            exact_query_terms = len(q_tokens & toks) / max(1, len(q_tokens))
            has_failure = 1.0 if toks & FAILURE_TERMS else 0.0
            has_update = 1.0 if toks & UPDATE_TERMS else 0.0
            recency = 0.5
            if candidates:
                newer_count = sum(
                    1 for other in candidates if _timestamp_key(other.timestamp) > _timestamp_key(cand.timestamp)
                )
                recency = 1.0 - min(1.0, newer_count / max(1, len(candidates) - 1))
            token_cost = min(1.0, estimate_tokens(text) / 900.0)
            score = 0.55 * relevance + 0.25 * exact_query_terms + 0.15 * recency - 0.05 * token_cost

            if cand.memory_index in stale_by_newer:
                newer = stale_by_newer[cand.memory_index]
                label = STALE
                confidence = 0.80
                usable = False
                reason = (
                    "A newer overlapping A-MEM candidate contains update/change language; "
                    f"newer_memory_index={newer.memory_index}."
                )
            elif has_failure and relevance >= self.support_threshold:
                label = WARNING
                confidence = min(0.90, 0.55 + relevance)
                usable = False
                reason = "Relevant failure/risk memory; include as warning rather than premise."
            elif score >= self.apply_threshold:
                label = APPLY
                confidence = min(0.95, 0.45 + score)
                usable = True
                reason = "Relevant to the current question/retrieval query and no newer contradiction was detected."
            elif score >= self.support_threshold:
                label = SUPPORT
                confidence = min(0.80, 0.35 + score)
                usable = False
                reason = "Partially relevant; keep only as background."
            elif has_update and relevance > 0.10:
                label = UNCERTAIN
                confidence = 0.45
                usable = False
                reason = "Update-like memory with weak match; keep uncertain unless corroborated."
            else:
                label = IRRELEVANT
                confidence = max(0.05, 0.40 - score)
                usable = False
                reason = "Low estimated applicability for this A-MEM query."

            decisions.append(
                GateDecision(
                    local_id=cand.local_id,
                    memory_index=cand.memory_index,
                    label=label,
                    usable_as_premise=usable,
                    confidence=round(float(confidence), 4),
                    reason=reason,
                    scores={
                        "relevance": float(relevance),
                        "query_term_coverage": float(exact_query_terms),
                        "recency": float(recency),
                        "has_failure_terms": has_failure,
                        "has_update_terms": has_update,
                        "token_cost": float(token_cost),
                        "heuristic_score": float(score),
                    },
                )
            )
        return decisions

    def _detect_stale_candidates(self, candidates: Sequence[AMemCandidate]) -> Dict[int, AMemCandidate]:
        stale: Dict[int, AMemCandidate] = {}
        for old in candidates:
            old_toks = token_set(old.content)
            for newer in candidates:
                if newer.memory_index == old.memory_index:
                    continue
                if _timestamp_key(newer.timestamp) <= _timestamp_key(old.timestamp):
                    continue
                newer_toks = token_set(newer.content)
                if not (newer_toks & UPDATE_TERMS):
                    continue
                if overlap_coeff(old_toks, newer_toks) < self.stale_overlap_threshold:
                    continue
                stale[old.memory_index] = newer
                break
        return stale


class LLMJsonAMemApplicabilityGate(AMemApplicabilityGate):
    name = "llm"

    def __init__(
        self,
        llm_controller: Any,
        fallback_gate: Optional[AMemApplicabilityGate] = None,
        max_candidates: int = 32,
    ):
        self.llm_controller = llm_controller
        self.fallback_gate = fallback_gate or HeuristicAMemApplicabilityGate()
        self.max_candidates = max_candidates

    def judge(
        self,
        question: str,
        retrieval_query: str,
        candidates: Sequence[AMemCandidate],
    ) -> List[GateDecision]:
        limited = list(candidates)[: self.max_candidates]
        prompt = self._build_prompt(question, retrieval_query, limited)
        try:
            raw = self.llm_controller.llm.get_completion(prompt, temperature=0.0)
            parsed = self._parse_json(raw)
            return self._decisions_from_json(parsed, limited)
        except Exception as exc:
            fallback = self.fallback_gate.judge(question, retrieval_query, candidates)
            for decision in fallback:
                decision.reason = f"LLM gate failed; fallback heuristic used. Error: {exc!r}. {decision.reason}"
                decision.scores["llm_gate_failed"] = 1.0
            return fallback

    def _build_prompt(
        self,
        question: str,
        retrieval_query: str,
        candidates: Sequence[AMemCandidate],
    ) -> str:
        candidate_blocks = []
        for cand in candidates:
            candidate_blocks.append(
                "\n".join(
                    [
                        f"ID: {cand.local_id}",
                        f"Memory index: {cand.memory_index}",
                        f"Source: {cand.source}",
                        f"Timestamp: {cand.timestamp}",
                        f"Content: {cand.content[:900]}",
                        f"Context: {cand.context[:400]}",
                        f"Keywords: {cand.keywords}",
                        f"Tags: {cand.tags}",
                    ]
                )
            )
        labels = ", ".join([APPLY, SUPPORT, WARNING, STALE, CONTRADICTED, UNCERTAIN, IRRELEVANT])
        return f"""You are an applicability authorization gate for an A-MEM read-time pipeline.

A-MEM has already generated a retrieval query, retrieved seed memories, and expanded linked memories.
Your task is NOT to retrieve more memories. Your task is to decide how each retrieved memory may be used.

Question:
{question}

A-MEM retrieval query / keywords:
{retrieval_query}

Allowed labels:
- APPLY: can be used as a current premise for answering.
- SUPPORT: relevant background only, not enough to decide the answer alone.
- WARNING: failure/risk/constraint memory; include as caution, not as a factual premise.
- STALE: historical memory superseded by newer evidence.
- CONTRADICTED: contradicted by another retrieved memory.
- UNCERTAIN: potentially relevant but unsafe to rely on.
- IRRELEVANT: should be dropped.

Rules:
1. A memory can be semantically relevant but still not APPLY.
2. Prefer newer update memories over older memories they supersede.
3. Failure/risk memories should usually be WARNING.
4. If there is not enough evidence, use SUPPORT or UNCERTAIN rather than APPLY.
5. Return strict JSON only. Do not add markdown.

Retrieved A-MEM candidates:
{chr(10).join("-----\\n" + block for block in candidate_blocks)}

Return JSON exactly in this shape:
{{
  "decisions": [
    {{
      "id": "c0",
      "label": "{APPLY}",
      "usable_as_premise": true,
      "confidence": 0.0,
      "reason": "short reason"
    }}
  ]
}}

Every candidate ID must appear once. Labels must be one of: {labels}.
"""

    def _parse_json(self, text: str) -> Dict[str, Any]:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _decisions_from_json(
        self,
        parsed: Dict[str, Any],
        candidates: Sequence[AMemCandidate],
    ) -> List[GateDecision]:
        by_id = {c.local_id: c for c in candidates}
        raw_decisions = parsed.get("decisions", [])
        out: List[GateDecision] = []
        seen = set()
        allowed = {APPLY, SUPPORT, WARNING, STALE, CONTRADICTED, UNCERTAIN, IRRELEVANT}
        for item in raw_decisions:
            cid = str(item.get("id", ""))
            cand = by_id.get(cid)
            if not cand:
                continue
            label = str(item.get("label", UNCERTAIN)).upper()
            if label not in allowed:
                label = UNCERTAIN
            confidence = float(item.get("confidence", 0.5))
            usable = bool(item.get("usable_as_premise", label == APPLY))
            if label != APPLY:
                usable = False
            out.append(
                GateDecision(
                    local_id=cand.local_id,
                    memory_index=cand.memory_index,
                    label=label,
                    usable_as_premise=usable,
                    confidence=max(0.0, min(1.0, confidence)),
                    reason=str(item.get("reason", ""))[:700],
                    scores={"llm_confidence": max(0.0, min(1.0, confidence))},
                )
            )
            seen.add(cid)
        if len(seen) != len(candidates):
            fallback = self.fallback_gate.judge("", "", [c for c in candidates if c.local_id not in seen])
            out.extend(fallback)
        out.sort(key=lambda d: int(d.local_id[1:]) if d.local_id[1:].isdigit() else 10**9)
        return out


def _dedupe_decisions(
    candidates: Sequence[AMemCandidate],
    decisions: Sequence[GateDecision],
) -> List[tuple[AMemCandidate, GateDecision]]:
    by_local = {d.local_id: d for d in decisions}
    label_priority = {
        APPLY: 0,
        WARNING: 1,
        STALE: 2,
        CONTRADICTED: 2,
        SUPPORT: 3,
        UNCERTAIN: 4,
        IRRELEVANT: 5,
    }
    best: Dict[int, tuple[AMemCandidate, GateDecision]] = {}
    for cand in candidates:
        decision = by_local.get(cand.local_id)
        if decision is None:
            continue
        current = best.get(cand.memory_index)
        if current is None:
            best[cand.memory_index] = (cand, decision)
            continue
        _, old_decision = current
        key_new = (label_priority.get(decision.label, 99), -decision.confidence, cand.order)
        key_old = (label_priority.get(old_decision.label, 99), -old_decision.confidence, current[0].order)
        if key_new < key_old:
            best[cand.memory_index] = (cand, decision)
    return sorted(best.values(), key=lambda pair: pair[0].order)


def build_authorized_packet(
    candidates: Sequence[AMemCandidate],
    decisions: Sequence[GateDecision],
    token_budget: int = 3500,
    max_apply: int = 12,
    max_support: int = 6,
    max_warning: int = 6,
    max_invalidated: int = 8,
) -> str:
    sections: Dict[str, List[str]] = {
        "Applicable facts": [],
        "Support-only background": [],
        "Warnings / failure memories": [],
        "Invalidated historical premises": [],
        "Uncertain memories": [],
    }
    limits = {
        "Applicable facts": max_apply,
        "Support-only background": max_support,
        "Warnings / failure memories": max_warning,
        "Invalidated historical premises": max_invalidated,
        "Uncertain memories": 4,
    }
    label_to_section = {
        APPLY: "Applicable facts",
        SUPPORT: "Support-only background",
        WARNING: "Warnings / failure memories",
        STALE: "Invalidated historical premises",
        CONTRADICTED: "Invalidated historical premises",
        UNCERTAIN: "Uncertain memories",
    }
    for cand, dec in _dedupe_decisions(candidates, decisions):
        section = label_to_section.get(dec.label)
        if not section:
            continue
        if len(sections[section]) >= limits[section]:
            continue
        sections[section].append(
            "\n".join(
                [
                    f"- [idx={cand.memory_index} id={dec.local_id}] label={dec.label} confidence={dec.confidence:.2f}",
                    f"  time: {cand.timestamp}",
                    f"  content: {cand.content[:900]}",
                    f"  context: {cand.context[:350]}",
                    f"  keywords: {cand.keywords}",
                    f"  reason: {dec.reason[:500]}",
                ]
            )
        )
    lines = [
        "AUTHORIZED MEMORY PACKET",
        "Use Applicable facts as current answer premises.",
        "Use Support-only background only for interpretation.",
        "Use Warnings / failure memories only to avoid repeated mistakes or unsafe assumptions.",
        "Do not use Invalidated historical premises as current facts.",
        "If important information is Uncertain, state uncertainty rather than guessing.",
        "",
    ]
    for section_name, items in sections.items():
        lines.append(f"{section_name}:")
        lines.extend(items if items else ["- None"])
        lines.append("")
    packet = "\n".join(lines)
    if estimate_tokens(packet) <= token_budget:
        return packet
    # Keep section headers and trim item lists deterministically under budget.
    trimmed_sections = {name: list(items) for name, items in sections.items()}
    order = [
        "Support-only background",
        "Uncertain memories",
        "Invalidated historical premises",
        "Warnings / failure memories",
        "Applicable facts",
    ]
    for name in order:
        while trimmed_sections[name] and estimate_tokens(_render_packet_from_sections(trimmed_sections)) > token_budget:
            trimmed_sections[name].pop()
    return _render_packet_from_sections(trimmed_sections)


def _render_packet_from_sections(sections: Dict[str, List[str]]) -> str:
    lines = [
        "AUTHORIZED MEMORY PACKET",
        "Use Applicable facts as current answer premises.",
        "Use Support-only background only for interpretation.",
        "Use Warnings / failure memories only to avoid repeated mistakes or unsafe assumptions.",
        "Do not use Invalidated historical premises as current facts.",
        "If important information is Uncertain, state uncertainty rather than guessing.",
        "",
    ]
    for section_name, items in sections.items():
        lines.append(f"{section_name}:")
        lines.extend(items if items else ["- None"])
        lines.append("")
    return "\n".join(lines)


def _format_official_context_from_candidates(candidates: Sequence[AMemCandidate]) -> str:
    return "".join(candidate.render_official_block() for candidate in candidates)


def retrieve_amem_candidates(memory_system: Any, query: str, k: int) -> tuple[str, List[AMemCandidate], List[int]]:
    """Replicate official RobustAgenticMemorySystem.find_related_memories_raw.

    Returns the exact raw context string plus structured candidates for gating.
    """

    if not memory_system.memories:
        return "", [], []
    indices = memory_system.retriever.search(query, k)
    all_memories = list(memory_system.memories.values())
    candidates: List[AMemCandidate] = []
    order = 0
    for seed_idx in indices:
        seed_memory = all_memories[seed_idx]
        candidates.append(_candidate_from_memory(seed_memory, seed_idx, "seed", None, order))
        order += 1
        j = 0
        for neighbor in seed_memory.links:
            neighbor_memory = all_memories[neighbor]
            candidates.append(_candidate_from_memory(neighbor_memory, neighbor, "linked", seed_idx, order))
            order += 1
            if j >= k:
                break
            j += 1
    return _format_official_context_from_candidates(candidates), candidates, list(indices)


def _candidate_from_memory(memory: Any, memory_index: int, source: str, seed_index: Optional[int], order: int) -> AMemCandidate:
    return AMemCandidate(
        local_id=f"c{order}",
        memory_index=int(memory_index),
        source=source,
        seed_index=seed_index,
        order=order,
        timestamp=str(getattr(memory, "timestamp", "") or ""),
        content=str(getattr(memory, "content", "") or ""),
        context=str(getattr(memory, "context", "") or ""),
        keywords=list(getattr(memory, "keywords", []) or []),
        tags=list(getattr(memory, "tags", []) or []),
        links=list(getattr(memory, "links", []) or []),
    )


def build_official_answer_prompt(
    context: str,
    question: str,
    category: int,
    answer: str,
    temperature_c5: float,
    category5_options: Optional[tuple[str, str]] = None,
) -> tuple[str, float]:
    """Exact category-specific prompt logic from A-MEM test_advanced_robust.py."""

    if category == 5:
        if category5_options is None:
            answer_tmp = ("Not mentioned in the conversation", answer)
        else:
            answer_tmp = category5_options
        user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
        temperature = temperature_c5
    elif category == 2:
        user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
        temperature = 0.7
    elif category == 3:
        user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
        temperature = 0.7
    else:
        user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
        temperature = 0.7
    return user_prompt, temperature


def _load_official_modules(amem_repo: Path) -> Dict[str, Any]:
    repo = amem_repo.resolve()
    if not (repo / "memory_layer_robust.py").exists():
        raise FileNotFoundError(f"Not an A-MEM repo root: {repo}")
    sys.path.insert(0, str(repo))
    try:
        import test_advanced_robust  # type: ignore
        import load_dataset  # type: ignore
        import llm_text_parsers  # type: ignore
        import utils  # type: ignore
        import memory_layer_robust  # type: ignore

        return {
            "test_advanced_robust": test_advanced_robust,
            "load_dataset": load_dataset,
            "llm_text_parsers": llm_text_parsers,
            "utils": utils,
            "memory_layer_robust": memory_layer_robust,
        }
    finally:
        # Keep path available for imported module relative imports during runtime.
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))


def _ensure_official_metric_resources(force: bool = False) -> None:
    """Fail fast if official A-MEM metric dependencies are missing.

    A-MEM's utils.calculate_metrics calls nltk.word_tokenize for BLEU. Recent
    NLTK releases need the extra punkt_tab package, while the official repo only
    downloads punkt and wordnet. Checking this before memory construction avoids
    wasting a long build only to crash on the first QA metric.
    """

    global _OFFICIAL_METRIC_RESOURCES_READY
    if _OFFICIAL_METRIC_RESOURCES_READY and not force:
        return

    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError("Official A-MEM metrics require nltk. Install the official requirements first.") from exc

    def download(package: str) -> None:
        print(f"[init] downloading NLTK resource for official metrics: {package}", flush=True)
        ok = nltk.download(package, quiet=True)
        if not ok:
            print(f"[init] warning: nltk.download({package!r}) returned False", flush=True)

    if force:
        for package in ("punkt", "punkt_tab"):
            download(package)

    try:
        nltk.word_tokenize("A-MEM metric preflight.")
    except LookupError:
        for package in ("punkt", "punkt_tab"):
            download(package)
        try:
            nltk.word_tokenize("A-MEM metric preflight.")
        except LookupError as exc:
            raise RuntimeError(
                "NLTK tokenizer data is still missing after attempting downloads. "
                "Run in Colab once: import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
            ) from exc

    for package, resource in (
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            download(package)

    _OFFICIAL_METRIC_RESOURCES_READY = True
    print("[init] NLTK resources for official metrics are ready", flush=True)


def _calculate_official_metrics(official: Dict[str, Any], prediction: str, reference: str) -> Dict[str, float]:
    _ensure_official_metric_resources()
    try:
        return official["utils"].calculate_metrics(prediction, reference)
    except LookupError:
        print("[metrics] NLTK resource lookup failed during metric calculation; refreshing resources once", flush=True)
        _ensure_official_metric_resources(force=True)
        return official["utils"].calculate_metrics(prediction, reference)


def _patch_official_metric_runtime(official: Dict[str, Any]) -> None:
    """Keep official metrics, but avoid reloading BERTScore's model every answer."""

    utils = official["utils"]
    if getattr(utils, "_aamem_lab_metric_runtime_patched", False):
        return

    try:
        from bert_score import BERTScorer
    except ImportError:
        print("[init] bert-score is not installed; official utils will handle BERTScore errors", flush=True)
        return

    scorer_cache: Dict[str, Any] = {}

    def calculate_bert_scores_cached(prediction: str, reference: str) -> Dict[str, float]:
        try:
            scorer = scorer_cache.get("en")
            if scorer is None:
                print("[init] loading cached official BERTScore scorer: lang=en / roberta-large", flush=True)
                scorer = BERTScorer(lang="en", verbose=False)
                scorer_cache["en"] = scorer
            try:
                precision, recall, f1_score = scorer.score([prediction], [reference], verbose=False)
            except TypeError:
                precision, recall, f1_score = scorer.score([prediction], [reference])
            return {
                "bert_precision": precision.item(),
                "bert_recall": recall.item(),
                "bert_f1": f1_score.item(),
            }
        except Exception as exc:
            print(f"Error calculating BERTScore: {exc}", flush=True)
            return {
                "bert_precision": 0.0,
                "bert_recall": 0.0,
                "bert_f1": 0.0,
            }

    utils.calculate_bert_scores = calculate_bert_scores_cached
    utils._aamem_lab_metric_runtime_patched = True
    print("[init] patched official BERTScore metric to reuse one cached scorer", flush=True)


def _patch_hf_backend(official: Dict[str, Any], max_new_tokens: int) -> None:
    """Patch official robust modules so backend='hf' works without changing A-MEM logic."""

    memory_layer = official["memory_layer_robust"]
    test_module = official["test_advanced_robust"]
    original_cls = getattr(memory_layer, "_aamem_original_robust_llm_controller", None)
    if original_cls is None:
        original_cls = memory_layer.RobustLLMController
        memory_layer._aamem_original_robust_llm_controller = original_cls

    class PatchedRobustLLMController(HFCompatibleRobustLLMController):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(
                *args,
                original_cls=original_cls,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

    memory_layer.RobustLLMController = PatchedRobustLLMController
    test_module.RobustLLMController = PatchedRobustLLMController


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def _preview_text(text: str, max_chars: int = 180) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _ensure_agent_memory_loaded(
    official: Dict[str, Any],
    amem_repo: Path,
    sample: Any,
    sample_idx: int,
    model: str,
    backend: str,
    retrieve_k: int,
    temperature_c5: float,
    sglang_host: str,
    sglang_port: int,
    cache_dir: Optional[Path],
    show_progress: bool = True,
) -> Any:
    agent_cls = official["test_advanced_robust"].RobustAdvancedMemAgent
    agent = agent_cls(model, backend, retrieve_k, temperature_c5, sglang_host, sglang_port)
    if cache_dir is None:
        cache_dir = amem_repo / f"cached_memories_robust_{backend}_{model}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    memory_cache_file = cache_dir / f"memory_cache_sample_{sample_idx}.pkl"
    retriever_cache_file = cache_dir / f"retriever_cache_sample_{sample_idx}.pkl"
    retriever_cache_embeddings_file = cache_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy"

    if memory_cache_file.exists():
        print(f"[cache] sample={sample_idx} memory_cache=hit path={memory_cache_file}", flush=True)
        with memory_cache_file.open("rb") as f:
            cached_memories = pickle.load(f)
        agent.memory_system.memories = cached_memories
        if retriever_cache_file.exists() and retriever_cache_embeddings_file.exists():
            print(f"[cache] sample={sample_idx} retriever_cache=hit", flush=True)
            agent.memory_system.retriever = agent.memory_system.retriever.load(
                str(retriever_cache_file), str(retriever_cache_embeddings_file)
            )
        else:
            print(f"[cache] sample={sample_idx} retriever_cache=miss -> rebuilding retriever from cached memories", flush=True)
            agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                cached_memories, "all-MiniLM-L6-v2"
            )
        return agent

    total_turns = sum(len(turns.turns) for turns in sample.conversation.sessions.values())
    print(
        f"[cache] sample={sample_idx} memory_cache=miss -> building A-MEM notes from {total_turns} turns",
        flush=True,
    )
    build_progress = _make_progress_bar(total_turns, desc=f"build sample {sample_idx}", unit="turn") if show_progress else None
    built_turns = 0
    for _, turns in sample.conversation.sessions.items():
        for turn in turns.turns:
            turn_datetime = turns.date_time
            conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
            agent.add_memory(conversation_tmp, time=turn_datetime)
            built_turns += 1
            if build_progress is not None:
                build_progress.update(1)
                build_progress.set_postfix(memories=len(agent.memory_system.memories))
            elif built_turns == 1 or built_turns % 10 == 0 or built_turns == total_turns:
                print(
                    f"[build] sample={sample_idx} turns={built_turns}/{total_turns} "
                    f"memories={len(agent.memory_system.memories)}",
                    flush=True,
                )
    if build_progress is not None:
        build_progress.close()

    with memory_cache_file.open("wb") as f:
        pickle.dump(agent.memory_system.memories, f)
    agent.memory_system.retriever.save(str(retriever_cache_file), str(retriever_cache_embeddings_file))
    print(f"[cache] sample={sample_idx} saved {len(agent.memory_system.memories)} memories to {memory_cache_file}", flush=True)
    return agent


def _aggregate_metric_rows(rows: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                grouped[key].append(float(value))
    out: Dict[str, Dict[str, float]] = {}
    for key, vals in grouped.items():
        out[key] = {
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "median": median(vals),
            "min": min(vals),
            "max": max(vals),
            "count": float(len(vals)),
        }
    return out


def _summary_mean(summary_block: Dict[str, Dict[str, float]], key: str) -> float:
    value = summary_block.get(key, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return 0.0


def _print_comparison_table(summary: Dict[str, Any]) -> None:
    answer_summary = summary.get("official_metric_summary_by_gate", {})
    context_summary = summary.get("context_summary_by_gate", {})
    headers = [
        "gate",
        "n",
        "EM",
        "F1",
        "ROUGE-L",
        "BLEU-1",
        "BLEU-4",
        "BERT-F1",
        "METEOR",
        "SBERT",
        "ctx_tok",
    ]
    rows: List[List[str]] = []
    for gate, metrics in answer_summary.items():
        ctx = context_summary.get(gate, {})
        n = int(float(metrics.get("f1", {}).get("count", 0.0))) if isinstance(metrics.get("f1"), dict) else 0
        rows.append(
            [
                gate,
                str(n),
                f"{_summary_mean(metrics, 'exact_match'):.3f}",
                f"{_summary_mean(metrics, 'f1'):.3f}",
                f"{_summary_mean(metrics, 'rougeL_f'):.3f}",
                f"{_summary_mean(metrics, 'bleu1'):.3f}",
                f"{_summary_mean(metrics, 'bleu4'):.3f}",
                f"{_summary_mean(metrics, 'bert_f1'):.3f}",
                f"{_summary_mean(metrics, 'meteor'):.3f}",
                f"{_summary_mean(metrics, 'sbert_similarity'):.3f}",
                f"{_summary_mean(ctx, 'final_context_tokens'):.0f}",
            ]
        )
    if not rows:
        print("\nNo metric rows to summarize.", flush=True)
        return

    widths = [len(x) for x in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fmt(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    print("\n=== Final comparison table: official A-MEM paper metrics ===", flush=True)
    print(fmt(headers), flush=True)
    print("-+-".join("-" * width for width in widths), flush=True)
    for row in rows:
        print(fmt(row), flush=True)


def _label_counts(decisions: Sequence[GateDecision]) -> Dict[str, int]:
    counts = {label: 0 for label in [APPLY, SUPPORT, WARNING, STALE, CONTRADICTED, UNCERTAIN, IRRELEVANT]}
    for decision in decisions:
        counts[decision.label] = counts.get(decision.label, 0) + 1
    return counts


def _count_planned_questions(samples: Sequence[Any], allowed_categories: set[int], max_questions: Optional[int]) -> int:
    total = 0
    for sample in samples:
        for qa in sample.qa:
            if int(qa.category) not in allowed_categories:
                continue
            total += 1
            if max_questions is not None and total >= max_questions:
                return total
    return total


def _make_progress_bar(total: int, desc: str = "official A-MEM QA", unit: str = "qa") -> Any:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def evaluate_official_amem_with_gates(args: argparse.Namespace) -> Dict[str, Any]:
    amem_repo = Path(args.amem_repo).resolve()
    print("=== Official A-MEM robust read-time eval + optional Layer-2 gate ===", flush=True)
    print(f"[init] amem_repo={amem_repo}", flush=True)
    print("[init] importing official A-MEM modules...", flush=True)
    official = _load_official_modules(amem_repo)
    print("[init] official modules imported", flush=True)
    _ensure_official_metric_resources()
    _patch_official_metric_runtime(official)
    requested_backends = {args.backend, args.gate_backend or args.backend}
    if "hf" in requested_backends:
        print("[init] enabling local HuggingFace backend patch for official robust A-MEM", flush=True)
        _patch_hf_backend(official, max_new_tokens=args.hf_max_new_tokens)
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = amem_repo / dataset_path
    print(f"[init] loading dataset={dataset_path}", flush=True)
    samples = official["load_dataset"].load_locomo_dataset(dataset_path)
    print(f"[init] loaded raw samples={len(samples)}", flush=True)
    if args.ratio < 1.0:
        samples = samples[: max(1, int(len(samples) * args.ratio))]
        print(f"[init] ratio={args.ratio} -> using samples={len(samples)}", flush=True)
    else:
        print(f"[init] ratio={args.ratio} -> using all loaded samples", flush=True)

    gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    allowed_categories = {int(c) for c in args.categories.split(",") if c.strip()}
    rng = random.Random(args.random_seed)

    gate_objects: Dict[str, Optional[AMemApplicabilityGate]] = {}
    for gate_name in gates:
        print(f"[init] preparing gate={gate_name}", flush=True)
        if gate_name == "none":
            gate_objects[gate_name] = None
        elif gate_name == "heuristic":
            gate_objects[gate_name] = HeuristicAMemApplicabilityGate()
        elif gate_name == "llm":
            gate_backend = args.gate_backend or args.backend
            gate_model = args.gate_model or args.model
            llm_controller_cls = official["memory_layer_robust"].RobustLLMController
            llm_controller = llm_controller_cls(
                backend=gate_backend,
                model=gate_model,
                sglang_host=args.sglang_host,
                sglang_port=args.sglang_port,
            )
            gate_objects[gate_name] = LLMJsonAMemApplicabilityGate(llm_controller)
        else:
            raise ValueError(f"Unknown gate: {gate_name}. Use none, heuristic, llm.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"official_amem_gate_{_safe_model_name(args.model)}_{timestamp}"
    jsonl_path = output_dir / f"{tag}.jsonl"
    summary_path = output_dir / f"{tag}_summary.json"

    print(f"[config] amem_repo={amem_repo}", flush=True)
    print(f"[config] dataset={dataset_path}", flush=True)
    print(f"[config] samples={len(samples)} categories={sorted(allowed_categories)} max_questions={args.max_questions}", flush=True)
    print(f"[config] model={args.model} backend={args.backend} retrieve_k={args.retrieve_k}", flush=True)
    print(f"[config] gates={gates} packet_token_budget={args.packet_token_budget}", flush=True)
    print(f"[config] output_jsonl={jsonl_path}", flush=True)
    print(f"[config] output_summary={summary_path}", flush=True)
    jsonl_path.write_text("", encoding="utf-8")
    print("[config] JSONL rows will be streamed incrementally during the run", flush=True)

    planned_questions = _count_planned_questions(samples, allowed_categories, args.max_questions)
    progress_bar = None if args.no_progress else _make_progress_bar(planned_questions)
    print(f"[progress] planned_questions={planned_questions}", flush=True)

    rows: List[Dict[str, Any]] = []
    metric_rows_by_gate: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    context_rows_by_gate: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    category_metric_rows_by_gate: Dict[str, Dict[int, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))

    total_questions = 0
    for sample_idx, sample in enumerate(samples):
        print(f"\n[sample={sample_idx}] loading/building official A-MEM memory cache", flush=True)
        agent = _ensure_agent_memory_loaded(
            official=official,
            amem_repo=amem_repo,
            sample=sample,
            sample_idx=sample_idx,
            model=args.model,
            backend=args.backend,
            retrieve_k=args.retrieve_k,
            temperature_c5=args.temperature_c5,
            sglang_host=args.sglang_host,
            sglang_port=args.sglang_port,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            show_progress=not args.no_progress,
        )
        memory_count = len(getattr(agent.memory_system, "memories", {}))
        print(f"[sample={sample_idx}] memories={memory_count} qa_items={len(sample.qa)}", flush=True)
        for qa_idx, qa in enumerate(sample.qa):
            category = int(qa.category)
            if category not in allowed_categories:
                continue
            if args.max_questions is not None and total_questions >= args.max_questions:
                break
            total_questions += 1

            retrieval_query = agent.generate_query_llm(qa.question)
            raw_context, candidates, seed_indices = retrieve_amem_candidates(
                agent.memory_system, retrieval_query, args.retrieve_k
            )
            raw_context_tokens = estimate_tokens(raw_context)
            print(
                f"\n[qa] sample={sample_idx} qa={qa_idx} cat={category} "
                f"question={_preview_text(qa.question)}",
                flush=True,
            )
            print(
                f"[retrieve] query={_preview_text(retrieval_query)} "
                f"seed_indices={seed_indices} candidates_after_links={len(candidates)} "
                f"raw_ctx_tok={raw_context_tokens}",
                flush=True,
            )
            category5_options = None
            if category == 5:
                if rng.random() < 0.5:
                    category5_options = ("Not mentioned in the conversation", qa.final_answer or "")
                else:
                    category5_options = (qa.final_answer or "", "Not mentioned in the conversation")
            for gate_name, gate in gate_objects.items():
                if gate is None:
                    context = raw_context
                    decisions: List[GateDecision] = []
                else:
                    decisions = gate.judge(qa.question, retrieval_query, candidates)
                    context = build_authorized_packet(
                        candidates,
                        decisions,
                        token_budget=args.packet_token_budget,
                        max_apply=args.max_apply,
                        max_support=args.max_support,
                        max_warning=args.max_warning,
                        max_invalidated=args.max_invalidated,
                    )

                prompt, temperature = build_official_answer_prompt(
                    context=context,
                    question=qa.question,
                    category=category,
                    answer=qa.final_answer or "",
                    temperature_c5=args.temperature_c5,
                    category5_options=category5_options,
                )
                start = time.time()
                try:
                    raw_response = agent.memory_system.llm_controller.llm.get_completion(
                        prompt, temperature=temperature
                    )
                except Exception as exc:
                    raw_response = ""
                    response_error = repr(exc)
                else:
                    response_error = ""
                latency = time.time() - start
                prediction = official["llm_text_parsers"].parse_plain_text_answer(raw_response)
                metrics = _calculate_official_metrics(official, prediction, qa.final_answer) if qa.final_answer else {
                    "exact_match": 0,
                    "f1": 0.0,
                    "rouge1_f": 0.0,
                    "rouge2_f": 0.0,
                    "rougeL_f": 0.0,
                    "bleu1": 0.0,
                    "bleu2": 0.0,
                    "bleu3": 0.0,
                    "bleu4": 0.0,
                    "bert_f1": 0.0,
                    "meteor": 0.0,
                    "sbert_similarity": 0.0,
                }
                context_metrics = {
                    "raw_context_tokens": float(estimate_tokens(raw_context)),
                    "final_context_tokens": float(estimate_tokens(context)),
                    "candidate_count": float(len(candidates)),
                    "seed_count": float(len(seed_indices)),
                    "answer_latency_sec": float(latency),
                }
                metric_rows_by_gate[gate_name].append(metrics)
                context_rows_by_gate[gate_name].append(context_metrics)
                category_metric_rows_by_gate[gate_name][category].append(metrics)
                row = {
                    "sample_id": sample_idx,
                    "qa_idx": qa_idx,
                    "gate": gate_name,
                    "question": qa.question,
                    "retrieval_query": retrieval_query,
                    "reference": qa.final_answer,
                    "category": category,
                    "seed_indices": seed_indices,
                    "prediction": prediction,
                    "raw_response": raw_response,
                    "response_error": response_error,
                    "metrics": metrics,
                    "context_metrics": context_metrics,
                    "label_counts": _label_counts(decisions),
                    "gate_decisions": [asdict(d) for d in decisions],
                    "candidates": [asdict(c) for c in candidates],
                    "raw_context": raw_context if args.include_contexts else "",
                    "final_context": context if args.include_contexts else "",
                    "prompt": prompt if args.include_prompts else "",
                }
                rows.append(row)
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"[answer] sample={sample_idx} qa={qa_idx} gate={gate_name} "
                    f"em={metrics.get('exact_match', 0):.3f} "
                    f"f1={metrics.get('f1', 0):.3f} "
                    f"rougeL={metrics.get('rougeL_f', 0):.3f} "
                    f"bleu1={metrics.get('bleu1', 0):.3f} "
                    f"meteor={metrics.get('meteor', 0):.3f} "
                    f"sbert={metrics.get('sbert_similarity', 0):.3f} "
                    f"ctx_tok={context_metrics['final_context_tokens']:.0f} "
                    f"labels={row['label_counts']} "
                    f"pred={_preview_text(prediction, 120)} "
                    f"gold={_preview_text(qa.final_answer or '', 120)}",
                    flush=True,
                )
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(sample=sample_idx, qa=qa_idx, cat=category)
            else:
                print(f"[progress] completed={total_questions}/{planned_questions}", flush=True)
        if args.max_questions is not None and total_questions >= args.max_questions:
            break

    if progress_bar is not None:
        progress_bar.close()

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "run_config": {
            "amem_repo": str(amem_repo),
            "dataset": str(dataset_path),
            "model": args.model,
            "backend": args.backend,
            "retrieve_k": args.retrieve_k,
            "ratio": args.ratio,
            "max_questions": args.max_questions,
            "gates": gates,
            "packet_token_budget": args.packet_token_budget,
        },
        "official_metric_summary_by_gate": {
            gate: _aggregate_metric_rows(metric_rows) for gate, metric_rows in metric_rows_by_gate.items()
        },
        "context_summary_by_gate": {
            gate: _aggregate_metric_rows(metric_rows) for gate, metric_rows in context_rows_by_gate.items()
        },
        "official_metric_summary_by_gate_category": {
            gate: {
                str(category): _aggregate_metric_rows(metric_rows)
                for category, metric_rows in by_category.items()
            }
            for gate, by_category in category_metric_rows_by_gate.items()
        },
        "rows_path": str(jsonl_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote rows: {jsonl_path}")
    print(f"Wrote summary: {summary_path}")
    _print_comparison_table(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official A-MEM robust LoCoMo evaluation with optional Layer-2 applicability gates."
    )
    parser.add_argument("--amem-repo", default="../A-mem-main/A-mem-main", help="Path to official A-MEM repo root.")
    parser.add_argument("--dataset", default="data/locomo10.json", help="Dataset path, absolute or relative to A-MEM repo.")
    parser.add_argument("--backend", default="openai", choices=["openai", "ollama", "sglang", "vllm", "hf"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--gates", default="none,heuristic", help="Comma-separated: none,heuristic,llm")
    parser.add_argument("--gate-backend", default=None)
    parser.add_argument("--gate-model", default=None)
    parser.add_argument("--hf-max-new-tokens", type=int, default=768)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--categories", default="1,2,3,4,5")
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--temperature-c5", type=float, default=0.5)
    parser.add_argument("--sglang-host", default="http://localhost")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--packet-token-budget", type=int, default=3500)
    parser.add_argument("--max-apply", type=int, default=12)
    parser.add_argument("--max-support", type=int, default=6)
    parser.add_argument("--max-warning", type=int, default=6)
    parser.add_argument("--max-invalidated", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--output-dir", default="runs/official_amem_gate")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--include-prompts", action="store_true")
    parser.add_argument("--include-contexts", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar; keep textual logs.")
    args = parser.parse_args()
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("--ratio must be in (0, 1].")
    return args


def main() -> None:
    evaluate_official_amem_with_gates(parse_args())


if __name__ == "__main__":
    main()
