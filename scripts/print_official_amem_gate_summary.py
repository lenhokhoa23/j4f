from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


OFFICIAL_METRICS = [
    "exact_match",
    "f1",
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "bert_f1",
    "meteor",
    "sbert_similarity",
]

CONTEXT_METRICS = [
    "raw_context_tokens",
    "final_context_tokens",
    "candidate_count",
    "seed_count",
    "answer_latency_sec",
]


def _mean(block: Dict[str, Dict[str, float]], key: str) -> float:
    value = block.get(key, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return 0.0


def _count(block: Dict[str, Dict[str, float]], key: str = "f1") -> int:
    value = block.get(key, {})
    if isinstance(value, dict):
        return int(float(value.get("count", 0.0)))
    return 0


def _format_table(headers: List[str], rows: Iterable[List[str]]) -> str:
    rows = list(rows)
    widths = [len(x) for x in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fmt(row: List[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    out = [fmt(headers), "-+-".join("-" * width for width in widths)]
    out.extend(fmt(row) for row in rows)
    return "\n".join(out)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def print_summary(summary_path: Path, show_examples: int) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    answer_summary = summary.get("official_metric_summary_by_gate", {})
    context_summary = summary.get("context_summary_by_gate", {})

    print(f"Summary file: {summary_path}")
    print(f"Rows file: {summary.get('rows_path', '')}")
    print("\nOfficial A-MEM paper metrics by gate")
    headers = [
        "gate",
        "n",
        "EM",
        "F1",
        "R1",
        "R2",
        "RL",
        "B1",
        "B4",
        "BERT-F1",
        "METEOR",
        "SBERT",
    ]
    metric_rows = []
    for gate, metrics in answer_summary.items():
        metric_rows.append(
            [
                gate,
                str(_count(metrics)),
                f"{_mean(metrics, 'exact_match'):.3f}",
                f"{_mean(metrics, 'f1'):.3f}",
                f"{_mean(metrics, 'rouge1_f'):.3f}",
                f"{_mean(metrics, 'rouge2_f'):.3f}",
                f"{_mean(metrics, 'rougeL_f'):.3f}",
                f"{_mean(metrics, 'bleu1'):.3f}",
                f"{_mean(metrics, 'bleu4'):.3f}",
                f"{_mean(metrics, 'bert_f1'):.3f}",
                f"{_mean(metrics, 'meteor'):.3f}",
                f"{_mean(metrics, 'sbert_similarity'):.3f}",
            ]
        )
    print(_format_table(headers, metric_rows))

    print("\nContext/token metrics by gate")
    headers = ["gate", "raw_tok", "final_tok", "reduction_%", "candidates", "seeds", "latency_s"]
    context_rows = []
    for gate, metrics in context_summary.items():
        raw_tok = _mean(metrics, "raw_context_tokens")
        final_tok = _mean(metrics, "final_context_tokens")
        reduction = 0.0 if raw_tok <= 0 else 100.0 * (raw_tok - final_tok) / raw_tok
        context_rows.append(
            [
                gate,
                f"{raw_tok:.0f}",
                f"{final_tok:.0f}",
                f"{reduction:.1f}",
                f"{_mean(metrics, 'candidate_count'):.1f}",
                f"{_mean(metrics, 'seed_count'):.1f}",
                f"{_mean(metrics, 'answer_latency_sec'):.2f}",
            ]
        )
    print(_format_table(headers, context_rows))

    by_category = summary.get("official_metric_summary_by_gate_category", {})
    if by_category:
        print("\nF1 / ROUGE-L by category")
        rows = []
        for gate, cats in by_category.items():
            for category, metrics in sorted(cats.items(), key=lambda item: int(item[0])):
                rows.append(
                    [
                        gate,
                        str(category),
                        str(_count(metrics)),
                        f"{_mean(metrics, 'f1'):.3f}",
                        f"{_mean(metrics, 'rougeL_f'):.3f}",
                    ]
                )
        print(_format_table(["gate", "cat", "n", "F1", "ROUGE-L"], rows))

    rows_path = Path(summary.get("rows_path", ""))
    if rows_path and not rows_path.exists():
        rows_path = summary_path.parent / rows_path.name
    if show_examples > 0 and rows_path.exists():
        print(f"\nFirst {show_examples} examples")
        for row in _load_jsonl(rows_path)[:show_examples]:
            print("-" * 80)
            print(f"sample={row.get('sample_id')} qa={row.get('qa_idx')} gate={row.get('gate')} cat={row.get('category')}")
            print(f"question: {row.get('question')}")
            print(f"retrieval_query: {row.get('retrieval_query')}")
            print(f"gold: {row.get('reference')}")
            print(f"pred: {row.get('prediction')}")
            print(f"metrics: {row.get('metrics')}")
            print(f"context_metrics: {row.get('context_metrics')}")
            print(f"labels: {row.get('label_counts')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print comparison tables for official A-MEM gate runs.")
    parser.add_argument("summary_path", type=Path)
    parser.add_argument("--show-examples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_summary(args.summary_path, args.show_examples)


if __name__ == "__main__":
    main()
