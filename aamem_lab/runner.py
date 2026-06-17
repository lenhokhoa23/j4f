from __future__ import annotations

import argparse
from pathlib import Path

from .baselines import build_methods
from .datasets import load_cases
from .evaluation import print_summary, run_methods_on_cases, write_jsonl, write_summary


DEFAULT_ACTMEM = Path("../frontier_memory_repos/ActMem/dataset/ActMemEval.json")
DEFAULT_LOCOMO = Path("../A-mem-main/A-mem-main/data/locomo10.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AAMem retrieval/gating experiments.")
    parser.add_argument("--dataset", choices=["actmem", "locomo"], required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--limit", type=int, default=20, help="Case/sample limit. For LoCoMo, limits samples unless --limit-questions is set.")
    parser.add_argument("--limit-questions", type=int, default=None, help="LoCoMo questions per sample.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--methods",
        default="raw_topk,amem_box,aamem,oracle",
        help="Comma-separated: raw_topk,amem_box,aamem,oracle",
    )
    parser.add_argument("--out-dir", default="runs/phase0")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def default_path(dataset: str) -> Path:
    here = Path(__file__).resolve().parents[1]
    if dataset == "actmem":
        return (here / DEFAULT_ACTMEM).resolve()
    if dataset == "locomo":
        return (here / DEFAULT_LOCOMO).resolve()
    raise ValueError(dataset)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).resolve() if args.data_path else default_path(args.dataset)
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = build_methods(method_names, k=args.k)

    if args.dataset == "locomo":
        cases = load_cases(
            args.dataset,
            data_path,
            limit=args.limit,
            limit_questions_per_sample=args.limit_questions,
        )
    else:
        cases = load_cases(args.dataset, data_path, limit=args.limit)

    tag = args.tag or f"{args.dataset}_n{len(cases)}_k{args.k}"
    out_dir = Path(args.out_dir)
    rows_path = out_dir / f"{tag}.jsonl"
    summary_path = out_dir / f"{tag}_summary.json"

    rows, metrics = run_methods_on_cases(cases, methods)
    write_jsonl(rows_path, rows)
    summary = write_summary(summary_path, metrics)
    print(f"Wrote rows: {rows_path}")
    print(f"Wrote summary: {summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()

