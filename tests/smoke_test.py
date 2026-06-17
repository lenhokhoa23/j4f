from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aamem_lab.baselines import build_methods
from aamem_lab.datasets import load_actmem_eval, load_locomo
from aamem_lab.evaluation import run_methods_on_cases


ROOT = Path(__file__).resolve().parents[2]


def test_actmem_smoke() -> None:
    path = ROOT / "frontier_memory_repos" / "ActMem" / "dataset" / "ActMemEval.json"
    cases = load_actmem_eval(path, limit=2)
    methods = build_methods(["raw_topk", "amem_box", "aamem", "oracle"], k=3)
    rows, metrics = run_methods_on_cases(cases, methods)
    assert rows
    assert metrics


def test_locomo_smoke() -> None:
    path = ROOT / "A-mem-main" / "A-mem-main" / "data" / "locomo10.json"
    cases = load_locomo(path, limit_samples=1, limit_questions_per_sample=2)
    methods = build_methods(["raw_topk", "aamem"], k=3)
    rows, metrics = run_methods_on_cases(cases, methods)
    assert rows
    assert metrics


if __name__ == "__main__":
    test_actmem_smoke()
    test_locomo_smoke()
    print("smoke tests passed")
