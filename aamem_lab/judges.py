from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .answer_metrics import normalize_answer, score_answer


@dataclass
class JudgeResult:
    provider: str
    model: str
    correct: float
    score: float
    reason: str
    latency_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseJudge:
    provider = "base"
    model = "none"

    def judge(self, question: str, gold_answer: str, predicted_answer: str) -> JudgeResult:
        raise NotImplementedError


class HeuristicJudge(BaseJudge):
    """Offline QA judge for smoke tests.

    This is deliberately conservative. It should not be used as a paper result,
    but it keeps the pipeline runnable without API keys.
    """

    provider = "heuristic"
    model = "token_f1_contains_v1"

    def judge(self, question: str, gold_answer: str, predicted_answer: str) -> JudgeResult:
        start = time.time()
        metrics = score_answer(gold_answer, predicted_answer)
        ngold = normalize_answer(gold_answer)
        npred = normalize_answer(predicted_answer)
        correct = 1.0 if (metrics.contains_gold or metrics.token_f1 >= 0.55) else 0.0
        if ngold and npred and ngold in npred:
            reason = "Gold answer appears in prediction."
        elif metrics.token_f1 >= 0.55:
            reason = "Token F1 is above heuristic correctness threshold."
        else:
            reason = "Prediction does not sufficiently match the gold answer."
        return JudgeResult(
            provider=self.provider,
            model=self.model,
            correct=correct,
            score=max(metrics.contains_gold, metrics.token_f1),
            reason=reason,
            latency_sec=time.time() - start,
            metadata={"token_f1": metrics.token_f1, "contains_gold": metrics.contains_gold},
        )


class OpenAICompatibleJudge(BaseJudge):
    provider = "openai_compatible"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_sec: int = 60,
    ):
        self.model = model or os.getenv("AAMEM_JUDGE_MODEL") or os.getenv("AAMEM_OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AAMEM_OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("AAMEM_OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        self.timeout_sec = timeout_sec
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY or AAMEM_OPENAI_API_KEY.")

    def _endpoint(self) -> str:
        url = self.base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    def judge(self, question: str, gold_answer: str, predicted_answer: str) -> JudgeResult:
        start = time.time()
        prompt = (
            "You are evaluating memory-agent QA. Decide whether the predicted answer is semantically "
            "correct with respect to the gold answer. Minor wording differences are allowed. "
            "Return only JSON with keys: correct (boolean), score (0 to 1), reason (short string).\n\n"
            f"Question: {question}\n"
            f"Gold answer: {gold_answer}\n"
            f"Predicted answer: {predicted_answer}\n"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 120,
        }
        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        text = parsed["choices"][0]["message"]["content"].strip()
        obj = self._parse_json(text)
        correct = 1.0 if bool(obj.get("correct")) else 0.0
        score = float(obj.get("score", correct))
        return JudgeResult(
            provider=self.provider,
            model=self.model,
            correct=correct,
            score=max(0.0, min(1.0, score)),
            reason=str(obj.get("reason", ""))[:500],
            latency_sec=time.time() - start,
            metadata={"raw": text, "usage": parsed.get("usage", {})},
        )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise


class NoJudge(BaseJudge):
    provider = "none"
    model = "none"

    def judge(self, question: str, gold_answer: str, predicted_answer: str) -> JudgeResult:
        return JudgeResult(
            provider=self.provider,
            model=self.model,
            correct=0.0,
            score=0.0,
            reason="No judge configured.",
            latency_sec=0.0,
        )


def build_judge(provider: str, model: Optional[str] = None) -> BaseJudge:
    if provider == "none":
        return NoJudge()
    if provider == "heuristic":
        return HeuristicJudge()
    if provider == "openai":
        return OpenAICompatibleJudge(model=model)
    raise ValueError(f"Unknown judge provider: {provider}")
