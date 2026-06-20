from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .official_amem_gate_runner import (
    HeuristicAMemApplicabilityGate,
    LLMJsonAMemApplicabilityGate,
    _aggregate_metric_rows,
    _calculate_official_metrics,
    _json_default,
    _label_counts,
    _load_official_modules,
    _make_progress_bar,
    _patch_hf_backend,
    _patch_official_metric_runtime,
    _preview_text,
    _print_comparison_table,
    _safe_model_name,
    build_authorized_packet,
    build_official_answer_prompt,
    estimate_tokens,
    retrieve_amem_candidates,
)


@dataclass(frozen=True)
class TurnCursor:
    session_id: int
    turn_index: int
    global_index: int
    dia_id: str

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.session_id, self.turn_index, self.global_index)


@dataclass
class PlannedQA:
    sample_idx: int
    qa_idx: int
    qa: Any
    cutoff: TurnCursor
    cutoff_source: str


def _iter_turns(sample: Any) -> List[tuple[TurnCursor, Any, Any]]:
    turns: List[tuple[TurnCursor, Any, Any]] = []
    global_index = 0
    for session_id in sorted(sample.conversation.sessions):
        session = sample.conversation.sessions[session_id]
        for turn_index, turn in enumerate(session.turns):
            cursor = TurnCursor(
                session_id=int(session_id),
                turn_index=int(turn_index),
                global_index=global_index,
                dia_id=str(turn.dia_id),
            )
            turns.append((cursor, session, turn))
            global_index += 1
    return turns


def _turn_lookup(turns: Sequence[tuple[TurnCursor, Any, Any]]) -> Dict[str, TurnCursor]:
    lookup: Dict[str, TurnCursor] = {}
    for cursor, _, turn in turns:
        lookup[cursor.dia_id] = cursor
        # Some code paths refer to only the numeric suffix; keep the exact id primary.
        if ":" in cursor.dia_id:
            lookup.setdefault(cursor.dia_id.split(":", 1)[1], cursor)
        if getattr(turn, "dia_id", None):
            lookup.setdefault(str(turn.dia_id), cursor)
    return lookup


def _parse_evidence_ids(evidence: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in evidence or []:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _qa_cutoff(
    qa: Any,
    turn_by_id: Dict[str, TurnCursor],
    last_cursor: TurnCursor,
    no_evidence_policy: str,
) -> Optional[tuple[TurnCursor, str]]:
    evidence_ids = _parse_evidence_ids(getattr(qa, "evidence", []) or [])
    if not evidence_ids:
        if no_evidence_policy == "skip":
            return None
        return last_cursor, "no_evidence_full_sample"

    cursors: List[TurnCursor] = []
    missing: List[str] = []
    for evidence_id in evidence_ids:
        cursor = turn_by_id.get(evidence_id)
        if cursor is None:
            missing.append(evidence_id)
            continue
        cursors.append(cursor)

    if not cursors:
        if no_evidence_policy == "skip":
            return None
        return last_cursor, "evidence_missing_full_sample:" + ",".join(missing[:8])

    cutoff = max(cursors, key=lambda c: c.key)
    suffix = "" if not missing else ";missing=" + ",".join(missing[:8])
    return cutoff, "max_evidence" + suffix


def _conversation_text(turn: Any) -> str:
    return "Speaker " + str(turn.speaker) + "says : " + str(turn.text)


def _planned_qas_for_sample(
    sample: Any,
    sample_idx: int,
    allowed_categories: set[int],
    remaining_budget: Optional[int],
    no_evidence_policy: str,
) -> tuple[List[PlannedQA], int]:
    turns = _iter_turns(sample)
    if not turns:
        return [], 0
    turn_by_id = _turn_lookup(turns)
    last_cursor = turns[-1][0]

    planned: List[PlannedQA] = []
    consumed = 0
    for qa_idx, qa in enumerate(sample.qa):
        if remaining_budget is not None and consumed >= remaining_budget:
            break
        category = int(qa.category)
        if category not in allowed_categories:
            continue
        cutoff_result = _qa_cutoff(qa, turn_by_id, last_cursor, no_evidence_policy)
        if cutoff_result is None:
            continue
        cutoff, cutoff_source = cutoff_result
        planned.append(
            PlannedQA(
                sample_idx=sample_idx,
                qa_idx=qa_idx,
                qa=qa,
                cutoff=cutoff,
                cutoff_source=cutoff_source,
            )
        )
        consumed += 1
    return planned, consumed


def _make_agent(
    official: Dict[str, Any],
    model: str,
    backend: str,
    retrieve_k: int,
    temperature_c5: float,
    sglang_host: str,
    sglang_port: int,
) -> Any:
    agent_cls = official["test_advanced_robust"].RobustAdvancedMemAgent
    return agent_cls(
        model=model,
        backend=backend,
        retrieve_k=retrieve_k,
        temperature_c5=temperature_c5,
        sglang_host=sglang_host,
        sglang_port=sglang_port,
    )


def _build_gate_objects(args: argparse.Namespace, official: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    gate_objects: Dict[str, Optional[Any]] = {}
    for gate_name in [g.strip() for g in args.gates.split(",") if g.strip()]:
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
            gate_objects[gate_name] = LLMJsonAMemApplicabilityGate(
                llm_controller,
                max_candidates=args.llm_gate_max_candidates,
                min_decision_coverage=args.llm_gate_min_coverage,
            )
        else:
            raise ValueError(f"Unknown gate: {gate_name}. Use none, heuristic, llm.")
    return gate_objects


def _count_planned(samples: Sequence[Any], args: argparse.Namespace, allowed_categories: set[int]) -> int:
    remaining = args.max_questions
    total = 0
    for sample_idx, sample in enumerate(samples):
        planned, consumed = _planned_qas_for_sample(
            sample=sample,
            sample_idx=sample_idx,
            allowed_categories=allowed_categories,
            remaining_budget=remaining,
            no_evidence_policy=args.no_evidence_policy,
        )
        total += len(planned)
        if remaining is not None:
            remaining -= consumed
            if remaining <= 0:
                break
    return total


def _summary_mean(summary_block: Dict[str, Dict[str, float]], key: str) -> float:
    value = summary_block.get(key, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return 0.0


def _print_prefix_comparison(summary: Dict[str, Any]) -> None:
    _print_comparison_table(summary)
    context_summary = summary.get("context_summary_by_gate", {})
    if not context_summary:
        return
    print("\n=== Prefix causal diagnostics ===", flush=True)
    for gate, block in context_summary.items():
        print(
            f"{gate}: "
            f"built_turns={_summary_mean(block, 'built_turn_count'):.1f} "
            f"candidate_count={_summary_mean(block, 'candidate_count'):.1f} "
            f"gate_decisions={_summary_mean(block, 'gate_decision_count'):.1f} "
            f"llm_failed={_summary_mean(block, 'llm_gate_failed_decisions'):.1f} "
            f"final_ctx_tok={_summary_mean(block, 'final_context_tokens'):.0f}",
            flush=True,
        )


def evaluate_prefix_amem_with_gates(args: argparse.Namespace) -> Dict[str, Any]:
    print("=== Causal/prefix official A-MEM read-time eval + optional Layer-2 gate ===", flush=True)
    print(
        "[mode] Memory is built incrementally and each QA is evaluated at max evidence cutoff. "
        "Future turns are not allowed to evolve/link/update memories for earlier QA.",
        flush=True,
    )

    amem_repo = Path(args.amem_repo).resolve()
    official = _load_official_modules(amem_repo)
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

    allowed_categories = {int(c) for c in args.categories.split(",") if c.strip()}
    gate_objects = _build_gate_objects(args, official)
    gates = list(gate_objects)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"official_amem_prefix_{_safe_model_name(args.model)}_{timestamp}"
    jsonl_path = output_dir / f"{tag}.jsonl"
    summary_path = output_dir / f"{tag}_summary.json"
    jsonl_path.write_text("", encoding="utf-8")

    planned_questions = _count_planned(samples, args, allowed_categories)
    progress_bar = None if args.no_progress else _make_progress_bar(planned_questions, desc="prefix A-MEM QA", unit="qa")

    print(f"[config] amem_repo={amem_repo}", flush=True)
    print(f"[config] dataset={dataset_path}", flush=True)
    print(
        f"[config] samples={len(samples)} categories={sorted(allowed_categories)} "
        f"max_questions={args.max_questions} no_evidence_policy={args.no_evidence_policy}",
        flush=True,
    )
    print(f"[config] model={args.model} backend={args.backend} retrieve_k={args.retrieve_k}", flush=True)
    print(
        f"[config] gates={gates} packet_token_budget={args.packet_token_budget} "
        f"llm_gate_max_candidates={args.llm_gate_max_candidates}",
        flush=True,
    )
    print(f"[config] output_jsonl={jsonl_path}", flush=True)
    print(f"[progress] planned_questions={planned_questions}", flush=True)

    rows: List[Dict[str, Any]] = []
    metric_rows_by_gate: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    context_rows_by_gate: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    category_metric_rows_by_gate: Dict[str, Dict[int, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))

    remaining = args.max_questions
    total_evaluated = 0
    for sample_idx, sample in enumerate(samples):
        planned, consumed = _planned_qas_for_sample(
            sample=sample,
            sample_idx=sample_idx,
            allowed_categories=allowed_categories,
            remaining_budget=remaining,
            no_evidence_policy=args.no_evidence_policy,
        )
        if not planned:
            continue
        if remaining is not None:
            remaining -= consumed

        turns = _iter_turns(sample)
        planned_by_global: Dict[int, List[PlannedQA]] = defaultdict(list)
        for item in planned:
            planned_by_global[item.cutoff.global_index].append(item)
        max_cutoff = max(item.cutoff.global_index for item in planned)

        print(
            f"\n[sample={sample_idx}] prefix build target_turn={max_cutoff + 1}/{len(turns)} "
            f"planned_qas={len(planned)}",
            flush=True,
        )
        agent = _make_agent(
            official=official,
            model=args.model,
            backend=args.backend,
            retrieve_k=args.retrieve_k,
            temperature_c5=args.temperature_c5,
            sglang_host=args.sglang_host,
            sglang_port=args.sglang_port,
        )

        build_progress = None
        if not args.no_build_progress:
            build_progress = _make_progress_bar(max_cutoff + 1, desc=f"prefix build sample {sample_idx}", unit="turn")

        built_turn_count = 0
        for cursor, session, turn in turns:
            if cursor.global_index > max_cutoff:
                break
            agent.add_memory(_conversation_text(turn), time=session.date_time)
            built_turn_count += 1
            if build_progress is not None:
                build_progress.update(1)
                build_progress.set_postfix(memories=len(agent.memory_system.memories))
            elif built_turn_count == 1 or built_turn_count % 10 == 0 or cursor.global_index == max_cutoff:
                print(
                    f"[build] sample={sample_idx} turn={built_turn_count}/{max_cutoff + 1} "
                    f"dia_id={cursor.dia_id} memories={len(agent.memory_system.memories)}",
                    flush=True,
                )

            due_qas = planned_by_global.get(cursor.global_index, [])
            for planned_qa in due_qas:
                qa = planned_qa.qa
                qa_idx = planned_qa.qa_idx
                category = int(qa.category)
                retrieval_query = agent.generate_query_llm(qa.question)
                raw_context, candidates, seed_indices = retrieve_amem_candidates(
                    agent.memory_system, retrieval_query, args.retrieve_k
                )
                raw_context_tokens = estimate_tokens(raw_context)
                print(
                    f"\n[qa-prefix] sample={sample_idx} qa={qa_idx} cat={category} "
                    f"cutoff={cursor.dia_id} built_turns={built_turn_count} "
                    f"question={_preview_text(qa.question)}",
                    flush=True,
                )
                print(
                    f"[retrieve-prefix] query={_preview_text(retrieval_query)} "
                    f"seed_indices={seed_indices} candidates_after_links={len(candidates)} "
                    f"raw_ctx_tok={raw_context_tokens}",
                    flush=True,
                )

                local_rng = random.Random(args.random_seed + sample_idx * 100000 + qa_idx)
                category5_options = None
                if category == 5:
                    if local_rng.random() < 0.5:
                        category5_options = ("Not mentioned in the conversation", qa.final_answer or "")
                    else:
                        category5_options = (qa.final_answer or "", "Not mentioned in the conversation")

                for gate_name, gate in gate_objects.items():
                    if gate is None:
                        context = raw_context
                        decisions: List[Any] = []
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
                        "raw_context_tokens": float(raw_context_tokens),
                        "final_context_tokens": float(estimate_tokens(context)),
                        "candidate_count": float(len(candidates)),
                        "seed_count": float(len(seed_indices)),
                        "answer_latency_sec": float(latency),
                        "built_turn_count": float(built_turn_count),
                        "cutoff_global_index": float(cursor.global_index),
                        "gate_decision_count": float(len(decisions)),
                        "llm_gate_failed_decisions": float(
                            sum(1 for d in decisions if getattr(d, "scores", {}).get("llm_gate_failed", 0.0))
                        ),
                        "llm_gate_missing_id_decisions": float(
                            sum(1 for d in decisions if getattr(d, "scores", {}).get("llm_gate_missing_id", 0.0))
                        ),
                    }
                    metric_rows_by_gate[gate_name].append(metrics)
                    context_rows_by_gate[gate_name].append(context_metrics)
                    category_metric_rows_by_gate[gate_name][category].append(metrics)
                    row = {
                        "sample_id": sample_idx,
                        "qa_idx": qa_idx,
                        "gate": gate_name,
                        "prefix_mode": "max_evidence_cutoff",
                        "cutoff_dia_id": cursor.dia_id,
                        "cutoff_session_id": cursor.session_id,
                        "cutoff_turn_index": cursor.turn_index,
                        "cutoff_global_index": cursor.global_index,
                        "cutoff_source": planned_qa.cutoff_source,
                        "built_turn_count": built_turn_count,
                        "memory_count": len(agent.memory_system.memories),
                        "question": qa.question,
                        "retrieval_query": retrieval_query,
                        "reference": qa.final_answer,
                        "category": category,
                        "evidence": list(getattr(qa, "evidence", []) or []),
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
                        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
                    print(
                        f"[answer-prefix] sample={sample_idx} qa={qa_idx} gate={gate_name} "
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

                total_evaluated += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(sample=sample_idx, qa=qa_idx, cutoff=cursor.dia_id)
                else:
                    print(f"[progress] completed={total_evaluated}/{planned_questions}", flush=True)

        if build_progress is not None:
            build_progress.close()
        if remaining is not None and remaining <= 0:
            break

    if progress_bar is not None:
        progress_bar.close()

    # Rewrite JSONL once at the end to guarantee a complete file even if rows were streamed.
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    summary = {
        "run_config": {
            "mode": "prefix_max_evidence_cutoff",
            "amem_repo": str(amem_repo),
            "dataset": str(dataset_path),
            "model": args.model,
            "backend": args.backend,
            "gate_backend": args.gate_backend or args.backend,
            "gate_model": args.gate_model or args.model,
            "retrieve_k": args.retrieve_k,
            "ratio": args.ratio,
            "max_questions": args.max_questions,
            "gates": gates,
            "packet_token_budget": args.packet_token_budget,
            "llm_gate_max_candidates": args.llm_gate_max_candidates,
            "no_evidence_policy": args.no_evidence_policy,
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
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"Wrote rows: {jsonl_path}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    _print_prefix_comparison(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run causal/prefix official A-MEM robust LoCoMo evaluation. "
            "Each QA is answered after max evidence cutoff instead of after full sample memory build."
        )
    )
    parser.add_argument("--amem-repo", default="../A-mem-main/A-mem-main", help="Path to official A-MEM repo root.")
    parser.add_argument("--dataset", default="data/locomo10.json", help="Dataset path, absolute or relative to A-MEM repo.")
    parser.add_argument("--backend", default="openai", choices=["openai", "ollama", "sglang", "vllm", "hf"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--gates", default="none,llm", help="Comma-separated: none,heuristic,llm")
    parser.add_argument("--gate-backend", default=None)
    parser.add_argument("--gate-model", default=None)
    parser.add_argument("--llm-gate-max-candidates", type=int, default=8)
    parser.add_argument("--llm-gate-min-coverage", type=float, default=0.8)
    parser.add_argument("--hf-max-new-tokens", type=int, default=768)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--categories", default="1,2,3,4,5")
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--temperature-c5", type=float, default=0.5)
    parser.add_argument("--sglang-host", default="http://localhost")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument(
        "--no-evidence-policy",
        choices=["skip", "full"],
        default="skip",
        help="How to handle QA items without evidence ids. 'skip' keeps prefix runs causal by default.",
    )
    parser.add_argument("--packet-token-budget", type=int, default=3500)
    parser.add_argument("--max-apply", type=int, default=12)
    parser.add_argument("--max-support", type=int, default=6)
    parser.add_argument("--max-warning", type=int, default=6)
    parser.add_argument("--max-invalidated", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--output-dir", default="runs/official_amem_prefix_gate")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--include-prompts", action="store_true")
    parser.add_argument("--include-contexts", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable QA tqdm progress bar.")
    parser.add_argument("--no-build-progress", action="store_true", help="Disable prefix memory build tqdm progress bar.")
    args = parser.parse_args()
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("--ratio must be in (0, 1].")
    if args.llm_gate_max_candidates <= 0:
        raise ValueError("--llm-gate-max-candidates must be positive.")
    if args.llm_gate_min_coverage <= 0.0 or args.llm_gate_min_coverage > 1.0:
        raise ValueError("--llm-gate-min-coverage must be in (0, 1].")
    return args


def main() -> None:
    evaluate_prefix_amem_with_gates(parse_args())


if __name__ == "__main__":
    main()
