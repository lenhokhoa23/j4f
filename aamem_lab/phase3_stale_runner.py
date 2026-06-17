from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .answerers import build_answerer
from .baselines import build_methods
from .metrics import score_result
from .model_presets import get_preset
from .paper_metrics import stale_style_dimension_metrics
from .serialization import method_result_to_dict
from .stale_suite import aggregate_stale_metrics, build_synthetic_stale_cases, score_stale_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: stale-memory stress test.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--noise-per-case", type=int, default=4)
    parser.add_argument("--dimensions", default="SR,PR,IPA", help="Comma-separated stale probe dimensions: SR,PR,IPA")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--methods",
        default="raw_topk,amem_box,aamem,aamem_stale_guard,oracle",
        help="Comma-separated method names.",
    )
    parser.add_argument("--provider", choices=["heuristic", "openai", "hf"], default="heuristic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-preset", default=None, help="Optional preset from aamem_lab.model_presets.")
    parser.add_argument("--include-prompts", action="store_true")
    parser.add_argument("--out-dir", default="runs/phase3_stale")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def _summarize(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["result"]["method"]].append(row)
    summary: Dict[str, Dict[str, float]] = {}
    for method, items in grouped.items():
        stale_summary = aggregate_stale_metrics([x["stale_metrics"] for x in items])
        stale_summary["evidence_recall"] = sum(float(x["memory_metrics"]["evidence_recall"]) for x in items) / len(items)
        stale_summary["evidence_precision"] = sum(float(x["memory_metrics"]["evidence_precision"]) for x in items) / len(items)
        stale_summary["noise_rate"] = sum(float(x["memory_metrics"]["noise_rate"]) for x in items) / len(items)
        stale_summary["avg_context_tokens"] = sum(float(x["result"]["token_estimate"]) for x in items) / len(items)
        stale_summary["avg_prompt_tokens_est"] = sum(float(x["answer"]["prompt_tokens_est"]) for x in items) / len(items)
        summary[method] = stale_summary
    return summary


def _print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    headers = [
        "method",
        "n",
        "fresh_mem",
        "stale_ctx",
        "stale_premise",
        "guard",
        "fresh_ans",
        "stale_ans",
        "probe_acc",
        "ans_f1",
        "tok",
    ]
    print("\t".join(headers))
    for method, vals in summary.items():
        print(
            "\t".join(
                [
                    method,
                    f"{vals.get('n', 0):.0f}",
                    f"{vals.get('fresh_memory_hit', 0):.3f}",
                    f"{vals.get('stale_context_leak', 0):.3f}",
                    f"{vals.get('stale_premise_leak', 0):.3f}",
                    f"{vals.get('stale_guard_rate', 0):.3f}",
                    f"{vals.get('fresh_answer_hit', 0):.3f}",
                    f"{vals.get('stale_answer_leak', 0):.3f}",
                    f"{vals.get('stale_probe_accuracy', 0):.3f}",
                    f"{vals.get('answer_token_f1', 0):.3f}",
                    f"{vals.get('avg_context_tokens', 0):.0f}",
                ]
            )
        )


def main() -> None:
    args = parse_args()
    dimensions = [d.strip().upper() for d in args.dimensions.split(",") if d.strip()]
    cases = build_synthetic_stale_cases(
        limit=args.limit,
        noise_per_case=args.noise_per_case,
        dimensions=dimensions,
    )
    methods = build_methods([m.strip() for m in args.methods.split(",") if m.strip()], k=args.k)
    if args.model_preset:
        preset = get_preset(args.model_preset)
        args.provider = preset.provider
        args.model = preset.model
    answerer = build_answerer(args.provider, model=args.model)

    rows: List[Dict] = []
    total = len(cases) * len(methods)
    step = 0
    for case in cases:
        for method in methods:
            step += 1
            result = method.run(case)
            answer = answerer.answer(case.query.query, result.context_text)
            stale_metrics = score_stale_result(case, result, answer.answer)
            memory_metrics = score_result(case, result)
            answer_row = {
                "provider": answer.provider,
                "model": answer.model,
                "answer": answer.answer,
                "latency_sec": answer.latency_sec,
                "prompt_tokens_est": answer.prompt_tokens_est,
                "completion_tokens_est": answer.completion_tokens_est,
                "metadata": answer.metadata,
            }
            if args.include_prompts:
                answer_row["prompt"] = answer.prompt
            row = {
                "case": {
                    "id": case.query.id,
                    "query": case.query.query,
                    "gold_answer": case.query.answer,
                    "fresh_memory_ids": case.query.metadata["fresh_memory_ids"],
                    "stale_memory_ids": case.query.metadata["stale_memory_ids"],
                    "stale_answer": case.query.metadata["stale_answer"],
                    "probe_dimension": case.query.metadata.get("probe_dimension"),
                },
                "result": method_result_to_dict(result, include_context=True),
                "answer": answer_row,
                "memory_metrics": memory_metrics.__dict__,
                "stale_metrics": stale_metrics,
                "run_config": {
                    "answer_provider": args.provider,
                    "answer_model": args.model or answer.model,
                    "model_preset": args.model_preset,
                    "k": args.k,
                    "dimensions": dimensions,
                },
            }
            rows.append(row)
            print(
                f"[{step}/{total}] {case.query.id} {result.method} "
                f"fresh_mem={stale_metrics['fresh_memory_hit']:.1f} "
                f"stale_premise={stale_metrics['stale_premise_leak']:.1f} "
                f"fresh_ans={stale_metrics['fresh_answer_hit']:.1f} "
                f"stale_ans={stale_metrics['stale_answer_leak']:.1f} "
                f"probe_acc={stale_metrics['stale_probe_accuracy']:.1f}"
            )

    tag = args.tag or f"synthetic_stale_n{len(cases)}_k{args.k}_{args.provider}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{tag}.jsonl"
    summary_path = out_dir / f"{tag}_summary.json"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = _summarize(rows)
    paper_summary = stale_style_dimension_metrics(rows)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "paper_summary": paper_summary}, f, ensure_ascii=False, indent=2)

    print(f"Wrote rows: {rows_path}")
    print(f"Wrote summary: {summary_path}")
    _print_summary(summary)


if __name__ == "__main__":
    main()
