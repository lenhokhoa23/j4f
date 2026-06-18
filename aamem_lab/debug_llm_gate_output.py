from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .official_amem_gate_runner import (
    LLMJsonAMemApplicabilityGate,
    _ensure_agent_memory_loaded,
    _load_official_modules,
    _patch_hf_backend,
    retrieve_amem_candidates,
)


def _cache_file(amem_repo: Path, backend: str, model: str, sample_idx: int, cache_dir: str | None) -> Path:
    root = Path(cache_dir) if cache_dir else amem_repo / f"cached_memories_robust_{backend}_{model}"
    return root / f"memory_cache_sample_{sample_idx}.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print raw LLM gate output for one official A-MEM QA item.")
    parser.add_argument("--amem-repo", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--qa-idx", type=int, default=0)
    parser.add_argument("--backend", default="hf", choices=["openai", "ollama", "sglang", "vllm", "hf"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gate-backend", default=None)
    parser.add_argument("--gate-model", default=None)
    parser.add_argument("--hf-max-new-tokens", type=int, default=768)
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--temperature-c5", type=float, default=0.5)
    parser.add_argument("--sglang-host", default="http://localhost")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--allow-build", action="store_true", help="Allow building sample memory if cache is absent.")
    parser.add_argument("--print-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    amem_repo = Path(args.amem_repo).resolve()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = amem_repo / dataset_path

    cache_file = _cache_file(amem_repo, args.backend, args.model, args.sample_idx, args.cache_dir)
    if not args.allow_build and not cache_file.exists():
        raise FileNotFoundError(
            f"Memory cache does not exist: {cache_file}\n"
            "Run official_amem_gate_runner first, or pass --allow-build to build the cache."
        )

    official = _load_official_modules(amem_repo)
    requested_backends = {args.backend, args.gate_backend or args.backend}
    if "hf" in requested_backends:
        _patch_hf_backend(official, max_new_tokens=args.hf_max_new_tokens)

    samples = official["load_dataset"].load_locomo_dataset(dataset_path)
    sample = samples[args.sample_idx]
    qa = sample.qa[args.qa_idx]
    agent = _ensure_agent_memory_loaded(
        official=official,
        amem_repo=amem_repo,
        sample=sample,
        sample_idx=args.sample_idx,
        model=args.model,
        backend=args.backend,
        retrieve_k=args.retrieve_k,
        temperature_c5=args.temperature_c5,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        show_progress=False,
    )

    retrieval_query = agent.generate_query_llm(qa.question)
    raw_context, candidates, seed_indices = retrieve_amem_candidates(
        agent.memory_system, retrieval_query, args.retrieve_k
    )

    gate_backend = args.gate_backend or args.backend
    gate_model = args.gate_model or args.model
    llm_controller_cls = official["memory_layer_robust"].RobustLLMController
    llm_controller = llm_controller_cls(
        backend=gate_backend,
        model=gate_model,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
    )
    gate = LLMJsonAMemApplicabilityGate(llm_controller, max_candidates=args.max_candidates)
    limited = candidates[: args.max_candidates]
    prompt = gate._build_prompt(qa.question, retrieval_query, limited)

    print("=== QA ===")
    print(f"sample_idx={args.sample_idx} qa_idx={args.qa_idx} category={qa.category}")
    print("question:", qa.question)
    print("gold:", qa.final_answer)
    print("retrieval_query:", retrieval_query)
    print("seed_indices:", seed_indices)
    print("candidate_count:", len(candidates), "limited:", len(limited))
    print("raw_context_chars:", len(raw_context))
    if args.print_prompt:
        print("\n=== LLM GATE PROMPT ===")
        print(prompt)

    print("\n=== RAW LLM GATE OUTPUT ===")
    raw = llm_controller.llm.get_completion(prompt, temperature=0.0)
    print(raw)

    print("\n=== PARSE RESULT ===")
    try:
        parsed = gate._parse_gate_output(raw)
        decisions = gate._decisions_from_json(parsed, limited)
    except Exception as exc:
        print("PARSE_FAILED:", repr(exc))
        return

    counts = Counter(decision.label for decision in decisions)
    print("label_counts:", dict(counts))
    for decision in decisions:
        print(json.dumps(asdict(decision), ensure_ascii=False))


if __name__ == "__main__":
    main()
