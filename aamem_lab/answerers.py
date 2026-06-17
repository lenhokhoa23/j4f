from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .text import estimate_tokens, overlap_coeff, token_set, tokenize, truncate_to_token_budget


SYSTEM_INSTRUCTION = (
    "You are a memory-grounded answerer. Answer only from the provided memory context. "
    "If the context is insufficient, say that the memory context is insufficient. "
    "Do not invent facts. Prefer current/fresh facts over old, stale, or warning facts."
)


@dataclass
class AnswerResult:
    provider: str
    model: str
    answer: str
    prompt: str
    latency_sec: float
    prompt_tokens_est: int
    completion_tokens_est: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_answer_prompt(question: str, memory_context: str, max_context_tokens: int = 3500) -> str:
    clipped_context = truncate_to_token_budget(memory_context, max_context_tokens)
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Question:\n{question}\n\n"
        f"Memory context:\n{clipped_context}\n\n"
        "Answer concisely. If useful, mention that a previous memory is stale or only background."
    )


class BaseAnswerer:
    provider: str = "base"
    model: str = "none"

    def answer(self, question: str, memory_context: str) -> AnswerResult:
        raise NotImplementedError


class HeuristicAnswerer(BaseAnswerer):
    """Fast offline answerer for pipeline smoke tests.

    This is not a strong language model. It chooses the highest-overlap sentence
    from the supplied memory context, with a small boost for current/update terms.
    Use it to verify logging, metrics, and stale traps before paying for model calls.
    """

    provider = "heuristic"
    model = "overlap_sentence_v1"

    def answer(self, question: str, memory_context: str) -> AnswerResult:
        start = time.time()
        prompt = build_answer_prompt(question, memory_context)
        answer = self._extract_sentence(question, memory_context)
        return AnswerResult(
            provider=self.provider,
            model=self.model,
            answer=answer,
            prompt=prompt,
            latency_sec=time.time() - start,
            prompt_tokens_est=estimate_tokens(prompt),
            completion_tokens_est=estimate_tokens(answer),
            metadata={"note": "offline extractive baseline, not an LLM"},
        )

    def _extract_sentence(self, question: str, context: str) -> str:
        if not context.strip() or "Applicable facts:\n- None" in context:
            return "The memory context is insufficient."
        q_tokens = token_set(question)
        raw_parts = re.split(r"(?<=[.!?])\s+|\n+", context)
        candidates = []
        for part in raw_parts:
            sentence = " ".join(part.split())
            if len(sentence) < 20:
                continue
            if sentence.startswith(("RAW RETRIEVED MEMORY", "AUTHORIZED MEMORY PACKET")):
                continue
            if sentence.startswith(("Use applicable", "Use support", "Do not use")):
                continue
            score = overlap_coeff(q_tokens, token_set(sentence))
            toks = set(tokenize(sentence))
            if toks & {"current", "currently", "now", "latest", "fresh", "updated"}:
                score += 0.12
            if toks & {"changed", "instead", "replaced", "moved", "no", "not"}:
                score += 0.08
            if "label=WARNING" in sentence or "Warnings / stale" in sentence:
                score -= 0.15
            if "reason:" in sentence:
                score -= 0.05
            candidates.append((score, sentence))
        if not candidates:
            return "The memory context is insufficient."
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        return best[:600]


class OpenAICompatibleAnswerer(BaseAnswerer):
    """OpenAI-compatible chat-completions provider using only stdlib urllib."""

    provider = "openai_compatible"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_sec: int = 60,
        max_output_tokens: int = 160,
        temperature: float = 0.0,
    ):
        self.model = model or os.getenv("AAMEM_OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AAMEM_OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("AAMEM_OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        self.timeout_sec = timeout_sec
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY or AAMEM_OPENAI_API_KEY.")

    def _endpoint(self) -> str:
        url = self.base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    def answer(self, question: str, memory_context: str) -> AnswerResult:
        start = time.time()
        prompt = build_answer_prompt(question, memory_context)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        answer = parsed["choices"][0]["message"]["content"].strip()
        usage = parsed.get("usage", {})
        return AnswerResult(
            provider=self.provider,
            model=self.model,
            answer=answer,
            prompt=prompt,
            latency_sec=time.time() - start,
            prompt_tokens_est=int(usage.get("prompt_tokens") or estimate_tokens(prompt)),
            completion_tokens_est=int(usage.get("completion_tokens") or estimate_tokens(answer)),
            metadata={"usage": usage, "endpoint": self._endpoint()},
        )


class HuggingFaceLocalAnswerer(BaseAnswerer):
    """Local transformers text-generation provider for Colab/GPU runs."""

    provider = "hf_local"

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        device_map: str = "auto",
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install transformers to use --provider hf.") from exc
        self.pipe = pipeline(
            "text-generation",
            model=model,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    def answer(self, question: str, memory_context: str) -> AnswerResult:
        start = time.time()
        prompt = build_answer_prompt(question, memory_context)
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        output = self.pipe(prompt, **kwargs)
        text = output[0]["generated_text"]
        answer = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
        # Some instruction models echo role markers; keep the answer bounded for logs.
        if "\nQuestion:" in answer:
            answer = answer.split("\nQuestion:", 1)[0].strip()
        return AnswerResult(
            provider=self.provider,
            model=self.model,
            answer=answer,
            prompt=prompt,
            latency_sec=time.time() - start,
            prompt_tokens_est=estimate_tokens(prompt),
            completion_tokens_est=estimate_tokens(answer),
            metadata={"device_map": self.device_map},
        )


def build_answerer(provider: str, model: Optional[str] = None) -> BaseAnswerer:
    if provider == "heuristic":
        return HeuristicAnswerer()
    if provider == "openai":
        return OpenAICompatibleAnswerer(model=model)
    if provider == "hf":
        if not model:
            raise ValueError("--model is required for --provider hf")
        return HuggingFaceLocalAnswerer(model=model)
    raise ValueError(f"Unknown answer provider: {provider}")
