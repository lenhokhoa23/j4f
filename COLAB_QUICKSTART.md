# Colab Quickstart

Core code không cần GPU. Colab chỉ cần khi bạn muốn chạy full dataset, gọi API model, hoặc tải HuggingFace model.

Notebook chính:

```text
notebooks/phase_runner.ipynb
```

Notebook này đã có section `Official A-MEM robust baseline + Layer-2 gate`. Section đó tự clone/pull repo này, clone/pull official A-MEM, chạy `--gates none,heuristic` hoặc `--gates none,heuristic,llm`, stream log từng sample/question/gate, rồi in bảng so sánh metric cuối.

## 1. Pull code từ GitHub

Notebook chính đã có cell tự clone/pull repo. Nếu muốn chạy thủ công trong Colab:

```python
from pathlib import Path
import os, subprocess, sys

REPO_URL = "https://github.com/lenhokhoa23/j4f.git"
PROJECT_DIR = Path("/content/j4f")

if (PROJECT_DIR / ".git").exists():
    subprocess.run(["git", "pull", "--ff-only"], cwd=PROJECT_DIR, check=True)
elif not PROJECT_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(PROJECT_DIR)], check=True)

os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
print(PROJECT_DIR)
```

Nếu dataset thật nằm ngoài repo, sửa `--data-path` cho đúng. Synthetic stale suite không cần dataset ngoài.

## 2. Chạy nhanh không cần model ngoài

```bash
python -m aamem_lab.phase2_answer_runner \
  --dataset actmem \
  --data-path ../frontier_memory_repos/ActMem/dataset/ActMemEval.json \
  --limit 10 \
  --k 5 \
  --model-preset smoke \
  --out-dir runs/phase2_answer \
  --tag colab_actmem_smoke
```

```bash
python -m aamem_lab.phase3_stale_runner \
  --limit 12 \
  --k 5 \
  --dimensions SR,PR,IPA \
  --model-preset smoke \
  --out-dir runs/phase3_stale \
  --tag colab_stale_smoke
```

Xem preset model nhỏ:

```bash
python scripts/show_model_presets.py
```

## 3. Chạy bằng model nhỏ 1B-8B trên Colab

```bash
pip install -q transformers accelerate
```

Các preset nên thử theo thứ tự:

```text
qwen2_5_1_5b -> qwen2_5_3b -> qwen2_5_7b
gemma2_2b
llama3_2_1b / llama3_2_3b nếu account HuggingFace có quyền
```

Ví dụ:

```bash
python -m aamem_lab.phase2_answer_runner \
  --dataset actmem \
  --data-path ../frontier_memory_repos/ActMem/dataset/ActMemEval.json \
  --limit 10 \
  --k 5 \
  --model-preset qwen2_5_1_5b \
  --include-prompts \
  --out-dir runs/phase2_answer \
  --tag colab_actmem_qwen1_5b
```

```bash
python -m aamem_lab.phase3_stale_runner \
  --limit 12 \
  --k 5 \
  --dimensions SR,PR,IPA \
  --model-preset qwen2_5_1_5b \
  --include-prompts \
  --out-dir runs/phase3_stale \
  --tag colab_stale_qwen1_5b
```

## 4. Chạy bằng OpenAI-compatible API

```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_KEY"
# Optional:
# os.environ["AAMEM_OPENAI_MODEL"] = "your-model"
# os.environ["AAMEM_OPENAI_BASE_URL"] = "https://api.openai.com/v1/chat/completions"
```

```bash
python -m aamem_lab.phase2_answer_runner \
  --dataset locomo \
  --data-path ../A-mem-main/A-mem-main/data/locomo10.json \
  --limit 1 \
  --limit-questions 10 \
  --k 5 \
  --model-preset openai_api \
  --include-prompts \
  --out-dir runs/phase2_answer \
  --tag colab_locomo_openai
```

Nếu Colab thiếu VRAM, giữ `--model-preset smoke` để debug pipeline trước.

## 5. File cần đọc sau khi chạy

- `runs/phase2_answer/*_summary.json`: answer-level summary.
- `runs/phase2_answer/*.jsonl`: từng prompt, context, answer, metric.
- `runs/phase3_stale/*_summary.json`: stale/fresh summary.
- `runs/phase3_stale/*.jsonl`: stale memory, fresh memory, gate decision, answer.
- `runs/official_amem_gate/*_summary.json`: official A-MEM paper metrics theo gate.
- `runs/official_amem_gate/*.jsonl`: từng LoCoMo QA, retrieval query, candidates, gate decision, prompt/context, prediction/gold và metric official.

Metric official A-MEM được in trong bảng gồm: exact match, F1, ROUGE-1/2/L, BLEU-1..4, BERTScore F1, METEOR và SBERT similarity. Bảng context kèm theo gồm raw/final context tokens, số candidate, số seed và latency.
