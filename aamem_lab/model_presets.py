from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelPreset:
    key: str
    provider: str
    model: Optional[str]
    judge_provider: str
    judge_model: Optional[str]
    notes: str
    expected_colab: str


MODEL_PRESETS: Dict[str, ModelPreset] = {
    "smoke": ModelPreset(
        key="smoke",
        provider="heuristic",
        model=None,
        judge_provider="heuristic",
        judge_model=None,
        notes="Fast offline pipeline check. Not a real LLM result.",
        expected_colab="CPU, seconds",
    ),
    "qwen2_5_1_5b": ModelPreset(
        key="qwen2_5_1_5b",
        provider="hf",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        judge_provider="heuristic",
        judge_model=None,
        notes="Cheap first real-model run; comparable in spirit to A-MEM's small local model setting.",
        expected_colab="T4/L4 OK, faster with 4-bit if you add quantization later",
    ),
    "qwen2_5_3b": ModelPreset(
        key="qwen2_5_3b",
        provider="hf",
        model="Qwen/Qwen2.5-3B-Instruct",
        judge_provider="heuristic",
        judge_model=None,
        notes="Good balance for LoCoMo subset and synthetic stale tests.",
        expected_colab="T4/L4 OK",
    ),
    "qwen2_5_7b": ModelPreset(
        key="qwen2_5_7b",
        provider="hf",
        model="Qwen/Qwen2.5-7B-Instruct",
        judge_provider="heuristic",
        judge_model=None,
        notes="Recommended upper small-model baseline before trying 9B/27B-scale paper models.",
        expected_colab="L4/A100 preferred; T4 may need quantization",
    ),
    "llama3_2_1b": ModelPreset(
        key="llama3_2_1b",
        provider="hf",
        model="meta-llama/Llama-3.2-1B-Instruct",
        judge_provider="heuristic",
        judge_model=None,
        notes="Matches the small Llama family used in A-MEM-style local experiments.",
        expected_colab="T4/L4 OK; may require HuggingFace access approval",
    ),
    "llama3_2_3b": ModelPreset(
        key="llama3_2_3b",
        provider="hf",
        model="meta-llama/Llama-3.2-3B-Instruct",
        judge_provider="heuristic",
        judge_model=None,
        notes="A stronger small Llama local baseline.",
        expected_colab="T4/L4 OK; may require HuggingFace access approval",
    ),
    "gemma2_2b": ModelPreset(
        key="gemma2_2b",
        provider="hf",
        model="google/gemma-2-2b-it",
        judge_provider="heuristic",
        judge_model=None,
        notes="Open small instruct baseline; useful if Llama access is blocked.",
        expected_colab="T4/L4 OK; may require HuggingFace access approval",
    ),
    "openai_api": ModelPreset(
        key="openai_api",
        provider="openai",
        model=None,
        judge_provider="openai",
        judge_model=None,
        notes="API run, use only when you have budget/key.",
        expected_colab="No GPU required",
    ),
}


def get_preset(key: str) -> ModelPreset:
    try:
        return MODEL_PRESETS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model preset: {key}. Available: {', '.join(MODEL_PRESETS)}") from exc


def presets_as_dict() -> list[dict]:
    return [preset.__dict__ for preset in MODEL_PRESETS.values()]
