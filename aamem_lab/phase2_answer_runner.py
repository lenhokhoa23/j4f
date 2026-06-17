from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

from .answer_metrics import score_answer
from .answerers import AnswerResult, build_answerer
from .baselines import build_methods
from .datasets import load_cases
from .judges import JudgeResult, build_judge
from .metrics import score_result
from .model_presets import get_preset
from .paper_metrics import actmem_style_metrics, aggregate_paper_metrics, locomo_style_metrics
from .runner import default_path
from .serialization import method_result_to_dict


def _answer_result_to_dict(result: AnswerResult, include_prompt: bool) -> Dict:
    row = {
        "provider": result.provider,
        "model": result.model,
        "answer": result.answer,
        "latency_sec": result.latency_sec,
        "prompt_tokens_est": result.prompt_tokens_est,
        "completion_tokens_est": result.completion_tokens_est,
        "metadata": result.metadata,
    }
    if include_prompt:
        row["prompt"] = result.prompt
    return row


def _judge_result_to_dict(result: JudgeResult) -> Dict:
    return {
        "provider": result.provider,
        "model": result.model,
        "correct": result.correct,
        "score": result.score,
        "reason": result.reason,
        "latency_sec": result.latency_sec,
        "metadata": result.metadata,
    }


def _summarize(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["result"]["method"]].append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for method, items in grouped.items():
        summary[method] = {
            "n": float(len(items)),
            "evidence_recall": mean(float(x["metrics"]["evidence_recall"]) for x in items),
            "evidence_precision": mean(float(x["metrics"]["evidence_precision"]) for x in items),
            "noise_rate": mean(float(x["metrics"]["noise_rate"]) for x in items),
            "avg_context_tokens": mean(float(x["result"]["token_estimate"]) for x in items),
            "answer_exact_match": mean(float(x["answer_metrics"]["exact_match"]) for x in items),
            "answer_contains_gold": mean(float(x["answer_metrics"]["contains_gold"]) for x in items),
            "answer_token_f1": mean(float(x["answer_metrics"]["token_f1"]) for x in items),
            "answer_rouge_l_f1": mean(float(x["answer_metrics"]["rouge_l_f1"]) for x in items),
            "avg_answer_latency_sec": mean(float(x["answer"]["latency_sec"]) for x in items),
            "judge_accuracy": mean(float(x["judge"]["correct"]) for x in items),
            "judge_score": mean(float(x["judge"]["score"]) for x in items),
            "avg_prompt_tokens_est": mean(float(x["answer"]["prompt_tokens_est"]) for x in items),
        }
    return summary


def _print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    headers = [
        "method",
        "n",
        "ev_recall",
        "ev_prec",
        "noise",
        "ctx_tok",
        "ans_f1",
        "judge_acc",
        "ans_contains",
        "prompt_tok",
    ]
    print("\t".join(headers))
    for method, vals in summary.items():
        print(
            "\t".join(
                [
                    method,
                    f"{vals['n']:.0f}",
                    f"{vals['evidence_recall']:.3f}",
                    f"{vals['evidence_precision']:.3f}",
                    f"{vals['noise_rate']:.3f}",
                    f"{vals['avg_context_tokens']:.0f}",
                    f"{vals['answer_token_f1']:.3f}",
                    f"{vals['judge_accuracy']:.3f}",
                    f"{vals['answer_contains_gold']:.3f}",
                    f"{vals['avg_prompt_tokens_est']:.0f}",
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: answer-level evaluation over memory packets.")
    parser.add_argument("--dataset", choices=["actmem", "locomo"], required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--methods", default="raw_topk,amem_box,aamem,oracle")
    parser.add_argument("--provider", choices=["heuristic", "openai", "hf"], default="heuristic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-provider", choices=["none", "heuristic", "openai"], default="heuristic")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--model-preset", default=None, help="Optional preset from aamem_lab.model_presets.")
    parser.add_argument("--include-prompts", action="store_true")
    parser.add_argument("--out-dir", default="runs/phase2_answer")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).resolve() if args.data_path else default_path(args.dataset)
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = build_methods(method_names, k=args.k)
    if args.model_preset:
        preset = get_preset(args.model_preset)
        args.provider = preset.provider
        args.model = preset.model
        args.judge_provider = preset.judge_provider
        args.judge_model = preset.judge_model
    answerer = build_answerer(args.provider, model=args.model)
    judge = build_judge(args.judge_provider, model=args.judge_model)

    if args.dataset == "locomo":
        cases = load_cases(
            args.dataset,
            data_path,
            limit=args.limit,
            limit_questions_per_sample=args.limit_questions,
        )
    else:
        cases = load_cases(args.dataset, data_path, limit=args.limit)

    rows: List[Dict] = []
    total = len(cases) * len(methods)
    step = 0
    for case in cases:
        for method in methods:
            step += 1
            result = method.run(case)
            memory_metrics = score_result(case, result)
            answer = answerer.answer(case.query.query, result.context_text)
            answer_metrics = score_answer(case.query.answer, answer.answer)
            judge_result = judge.judge(case.query.query, case.query.answer, answer.answer)
            memory_metrics_dict = memory_metrics.__dict__
            answer_metrics_dict = answer_metrics.__dict__
            judge_dict = _judge_result_to_dict(judge_result)
            if args.dataset == "actmem":
                paper_metrics = actmem_style_metrics(memory_metrics_dict, answer_metrics_dict, judge_dict)
            else:
                paper_metrics = locomo_style_metrics(memory_metrics_dict, answer_metrics_dict, judge_dict)
            row = {
                "case": {
                    "id": case.query.id,
                    "dataset": case.query.source_dataset,
                    "sample_id": case.query.sample_id,
                    "query": case.query.query,
                    "gold_answer": case.query.answer,
                    "gold_memory_ids": case.query.gold_memory_ids,
                    "category": case.query.category,
                },
                "result": method_result_to_dict(result, include_context=True),
                "answer": _answer_result_to_dict(answer, include_prompt=args.include_prompts),
                "judge": judge_dict,
                "metrics": memory_metrics_dict,
                "answer_metrics": answer_metrics_dict,
                "paper_metrics": paper_metrics,
                "run_config": {
                    "answer_provider": args.provider,
                    "answer_model": args.model or answer.model,
                    "judge_provider": args.judge_provider,
                    "judge_model": args.judge_model or judge_result.model,
                    "model_preset": args.model_preset,
                    "k": args.k,
                },
            }
            rows.append(row)
            print(
                f"[{step}/{total}] {case.query.id} {result.method} "
                f"ev_recall={memory_metrics.evidence_recall:.3f} "
                f"noise={memory_metrics.noise_rate:.3f} "
                f"ans_f1={answer_metrics.token_f1:.3f} "
                f"judge={judge_result.correct:.0f} "
                f"latency={answer.latency_sec:.2f}s"
            )

    tag = args.tag or f"{args.dataset}_n{len(cases)}_k{args.k}_{args.provider}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{tag}.jsonl"
    summary_path = out_dir / f"{tag}_summary.json"

    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = _summarize(rows)
    paper_summary = aggregate_paper_metrics(rows)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "paper_summary": paper_summary}, f, ensure_ascii=False, indent=2)

    print(f"Wrote rows: {rows_path}")
    print(f"Wrote summary: {summary_path}")
    _print_summary(summary)


if __name__ == "__main__":
    main()
