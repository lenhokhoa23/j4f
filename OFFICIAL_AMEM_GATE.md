# Official A-MEM + Layer-2 Gate Runner

Tài liệu này hướng dẫn chạy đúng read-time pipeline của official A-MEM robust evaluation, rồi bọc thêm Layer-2 Applicability Gate ở đúng vị trí:

```text
A-MEM query generation
  -> A-MEM retrieve top-k seeds
  -> A-MEM linked-memory expansion
  -> Layer-2 gate, optional
  -> official A-MEM category-specific answer prompt
  -> official A-MEM metrics
```

Nếu chạy trên Colab, mở `notebooks/phase_runner.ipynb` và chạy section `Official A-MEM robust baseline + Layer-2 gate`. Notebook đã tích hợp clone/pull code, setup official A-MEM, stream log realtime và in bảng so sánh cuối.

Điểm quan trọng:

```text
--gates none
```

là đường baseline không bọc gate. Nó giữ đúng logic robust A-MEM:

```text
generate_query_llm(question)
find_related_memories_raw(keywords, k)
official prompt theo category
LLM answer
utils.calculate_metrics
```

Gate chỉ được chèn sau khi A-MEM đã retrieve + expand linked memories, trước khi context được đưa vào prompt.

---

## 1. Chuẩn bị repo

Trong Colab hoặc máy local:

```bash
git clone https://github.com/lenhokhoa23/j4f.git
cd j4f
```

Clone official A-MEM cạnh repo này:

```bash
cd ..
git clone https://github.com/WujiangXu/AgenticMemory.git A-mem-main
cd A-mem-main
pip install -r requirements.txt
cd ../j4f
```

Nếu bạn đã có folder local giống workspace hiện tại:

```text
../A-mem-main/A-mem-main
```

thì có thể dùng luôn.

---

## 2. Chạy baseline không bọc gate

Lệnh này dùng đúng official A-MEM robust path, không gate:

```bash
python -m aamem_lab.official_amem_gate_runner \
  --amem-repo ../A-mem-main/A-mem-main \
  --dataset data/locomo10.json \
  --backend ollama \
  --model qwen2.5:3b \
  --gates none \
  --ratio 0.5 \
  --retrieve-k 10 \
  --output-dir runs/official_amem_gate \
  --tag amem_none_qwen3b_r50
```

Nếu muốn check gần nhất với script official:

```bash
python -m aamem_lab.official_amem_gate_runner \
  --amem-repo ../A-mem-main/A-mem-main \
  --dataset data/locomo10.json \
  --backend openai \
  --model gpt-4o-mini \
  --gates none \
  --ratio 1.0 \
  --retrieve-k 10 \
  --output-dir runs/official_amem_gate \
  --tag amem_none_gpt4omini_full
```

Ghi chú:

- Official script random hóa thứ tự option ở category 5. Runner này dùng `--random-seed` để so gate công bằng. Nếu muốn giảm nhiễu khi so với official script gốc, có thể chạy `--categories 1,2,3,4` trước.
- Official metrics có BERTScore/METEOR/ROUGE/BLEU nên lần đầu có thể tải model hoặc NLTK data.
- Nếu Colab lỗi vLLM/CUDA như `libcudart.so.13`, dùng `--backend hf` để gọi model HuggingFace trực tiếp bằng `transformers`.

Ví dụ Colab không cần vLLM:

```bash
python -m aamem_lab.official_amem_gate_runner \
  --amem-repo ../AgenticMemory \
  --dataset ../AgenticMemory/data/locomo10.json \
  --backend hf \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gates none,heuristic \
  --ratio 0.1 \
  --max-questions 30 \
  --retrieve-k 10 \
  --hf-max-new-tokens 768 \
  --output-dir runs/official_amem_gate \
  --tag amem_none_heuristic_hf_qwen15b_n30
```

---

## 3. Chạy baseline + heuristic gate

Đây là cách 1: gate rule-based để debug đúng wrapper, không phải final method.

```bash
python -m aamem_lab.official_amem_gate_runner \
  --amem-repo ../A-mem-main/A-mem-main \
  --dataset data/locomo10.json \
  --backend ollama \
  --model qwen2.5:3b \
  --gates none,heuristic \
  --ratio 0.5 \
  --retrieve-k 10 \
  --packet-token-budget 3500 \
  --output-dir runs/official_amem_gate \
  --tag amem_none_vs_heuristic_qwen3b_r50 \
  --include-contexts
```

So sánh chính:

```text
none      = official A-MEM robust context
heuristic = official A-MEM context đã qua authorized packet
```

---

## 4. Chạy baseline + heuristic + LLM gate

Đây là cách 2: LLM JSON gate. Gate dùng cùng backend/model hoặc backend/model riêng.

Ví dụ dùng Ollama cho cả answer và gate:

```bash
python -m aamem_lab.official_amem_gate_runner \
  --amem-repo ../A-mem-main/A-mem-main \
  --dataset data/locomo10.json \
  --backend ollama \
  --model qwen2.5:3b \
  --gates none,heuristic,llm \
  --gate-backend ollama \
  --gate-model qwen2.5:3b \
  --ratio 0.5 \
  --retrieve-k 10 \
  --packet-token-budget 3500 \
  --output-dir runs/official_amem_gate \
  --tag amem_none_heuristic_llm_qwen3b_r50 \
  --include-contexts
```

Ví dụ dùng vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 30000 \
  --dtype float16 \
  --enforce-eager \
  --max-model-len 8192

python -m aamem_lab.official_amem_gate_runner \
  --amem-repo ../A-mem-main/A-mem-main \
  --dataset data/locomo10.json \
  --backend vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --gates none,heuristic,llm \
  --gate-backend vllm \
  --gate-model Qwen/Qwen2.5-3B-Instruct \
  --ratio 0.5 \
  --retrieve-k 10 \
  --sglang-port 30000 \
  --output-dir runs/official_amem_gate \
  --tag amem_none_heuristic_llm_vllm_qwen3b_r50
```

---

## 5. Subset to hơn

Các mức khuyến nghị:

```bash
# nhanh, debug pipeline
--max-questions 20

# subset vừa, đáng đọc log
--ratio 0.3

# subset lớn hơn
--ratio 0.5

# full locomo10
--ratio 1.0
```

Với `none,heuristic,llm`, số lần answer/gate call tăng mạnh. Nếu model local chậm, bắt đầu bằng:

```text
--max-questions 30 --gates none,heuristic
```

rồi mới thêm `llm`.

---

## 6. Output

Runner ghi:

```text
runs/official_amem_gate/<tag>.jsonl
runs/official_amem_gate/<tag>_summary.json
```

Trong summary có:

```json
{
  "official_metric_summary_by_gate": {
    "none": {},
    "heuristic": {},
    "llm": {}
  },
  "context_summary_by_gate": {
    "none": {},
    "heuristic": {},
    "llm": {}
  },
  "official_metric_summary_by_gate_category": {}
}
```

Các metric official A-MEM được giữ:

```text
exact_match
f1
rouge1_f
rouge2_f
rougeL_f
bleu1-4
bert_f1
meteor
sbert_similarity
```

Các metric context phụ để đọc tác động của gate:

```text
raw_context_tokens
final_context_tokens
candidate_count
seed_count
answer_latency_sec
label_counts
```

---

## 7. Cách đọc kết quả

Câu hỏi chính:

```text
A-MEM + gate có giữ hoặc tăng answer metric không?
Nó có giảm final_context_tokens không?
Nó có label hợp lý các memory stale/failure/background không?
```

Nếu `none` tốt hơn nhiều:

```text
gate quá bảo thủ hoặc packet làm mất evidence
```

Nếu `heuristic` giảm token nhưng answer metric giảm:

```text
cần LLM gate hoặc learned gate
```

Nếu `llm` giữ metric và giảm token/noise:

```text
Layer-2 gate có tín hiệu tốt để phát triển thành verifier học được
```

---

## 8. Đảm bảo không lệch A-MEM

Runner cố ý giữ các bước official:

```text
1. add_memory: "Speaker " + turn.speaker + "says : " + turn.text
2. generate_query_llm(question)
3. find_related_memories_raw behavior:
   - retrieve seed indices
   - append each seed memory
   - append each linked neighbor from memory.links
4. prompt template theo category 1-5
5. parse_plain_text_answer
6. utils.calculate_metrics
```

Điểm duy nhất thay đổi khi bật gate:

```text
context = raw_context
```

được thay bằng:

```text
context = AUTHORIZED MEMORY PACKET
```

sau khi A-MEM đã retrieve + expand xong.
