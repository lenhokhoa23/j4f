# AAMem Lab: Applicability-Aware Memory Agent

Folder này là bộ test nhanh nhưng có cấu trúc để kiểm tra ý tưởng:

> Retrieval relevance không đồng nghĩa với memory applicability.

GitHub target:

```text
https://github.com/lenhokhoa23/j4f.git
```

Notebook Colab chính:

```text
notebooks/phase_runner.ipynb
```

Cell đầu của notebook sẽ clone/pull repo trên rồi chạy code từ bản mới nhất. Notebook này đã có section riêng để clone/pull official A-MEM, chạy baseline `none`, chạy gate `heuristic`/`llm`, stream log từng bước và in bảng so sánh cuối.

Official A-MEM + Layer-2 gate:

```text
OFFICIAL_AMEM_GATE.md
```

Runner này dùng đúng read-time pipeline robust của official A-MEM để so:

```text
--gates none              # không bọc gate, giữ A-MEM baseline
--gates none,heuristic    # baseline + heuristic Layer-2 gate
--gates none,heuristic,llm
```

Bảng cuối dùng metric official A-MEM từ `utils.calculate_metrics`: exact match, F1, ROUGE-1/2/L, BLEU-1..4, BERTScore F1, METEOR và SBERT similarity. Bảng phụ in token/context metric để xem gate giảm prompt hay làm tụt/tăng chất lượng trả lời.

Nói ngắn gọn: một memory có thể được retriever kéo lên top-k vì giống câu hỏi về mặt từ vựng/ngữ nghĩa, nhưng nó vẫn có thể không nên đưa vào prompt vì nó cũ, sai ngữ cảnh, mâu thuẫn, chỉ là background, hoặc quá tốn token so với giá trị thực sự. AAMem thêm một lớp gate sau retrieval và trước prompt để quyết định memory nào được phép dùng như premise.

## Ta đang đánh vào điểm nào?

Các hệ memory agent hiện tại thường làm pipeline kiểu:

1. Lưu memory thành các record.
2. Query đến thì retrieve top-k.
3. Có thể expand thêm neighbor/linked memory.
4. Đưa cả cụm memory vào prompt.
5. LLM tự xoay sở.

Điểm yếu là bước 3-4: nếu top-k sai hoặc neighbor nhiều nhiễu, prompt phình token và LLM dễ dùng sai memory. A-MEM-style retrieval càng rõ vấn đề này: top-k seed được kéo lên, rồi các linked/neighborhood memory được append thêm. Cơ chế này tốt cho recall nhưng chưa có kiểm tra “memory này có được phép dùng cho câu hỏi hiện tại không?”.

Ý tưởng trong repo này không thay retriever ngay. Nó thêm một tầng:

```text
query
  -> retriever lấy candidate memory
  -> optional box/neighborhood expansion
  -> applicability gate gán nhãn từng memory
  -> authorized memory packet
  -> LLM answerer hoặc evaluator
```

Các nhãn gate hiện tại:

- `APPLY`: được dùng như premise trực tiếp.
- `SUPPORT_ONLY`: chỉ dùng làm nền, không được kết luận trực tiếp.
- `WARNING`: có ích nhưng có dấu hiệu stale/mâu thuẫn, chỉ đưa vào như cảnh báo.
- `STALE`: ký ức cũ hoặc bị thay thế, dùng cho phase sau.
- `CONTRADICTED`: bị memory khác phản bác, dùng cho phase sau.
- `UNCERTAIN`: liên quan nhưng chưa đủ chắc.
- `IRRELEVANT`: loại khỏi packet.

## Dataset dùng để test claim

Không nên chọn dataset chỉ vì nó nổi tiếng. Claim đánh vào đâu thì dataset phải đo đúng điểm đó.

### ActMemEval

Dùng cho Phase 0 vì mỗi câu hỏi có `answer_session_ids`. Đây là label tốt để đo:

- method có giữ được evidence session không;
- method có kéo quá nhiều session nhiễu không;
- token cost tăng/giảm thế nào;
- A-MEM-style box expansion có tăng recall nhưng tăng noise không.

Lệnh nhanh:

```powershell
python -m aamem_lab.runner --dataset actmem --limit 20 --k 5 --out-dir runs/phase0_actmem
```

### LoCoMo local subset

Dùng để test long-term personal memory:

- memory ở mức dialogue turn;
- câu hỏi có evidence `D1:3`, `D8:6; D9:17`, hoặc evidence rỗng;
- có câu hỏi multi-hop, temporal, adversarial/unanswerable;
- dễ nhìn prompt packet cụ thể.

Lệnh nhanh:

```powershell
python -m aamem_lab.runner --dataset locomo --limit 1 --limit-questions 20 --k 5 --out-dir runs/phase0_locomo
```

## Method đã implement

### `raw_topk`

BM25 top-k thuần. Đây là baseline cơ bản:

```text
query -> BM25 -> top-k memory -> raw context
```

Nó đo xem retriever thô đã đủ chưa.

### `amem_box`

Proxy cho hành vi retrieval-time của A-MEM:

```text
query -> top-k seed -> build token-neighbor links -> append neighbors -> raw context
```

Đây chưa phải full official A-MEM end-to-end. Nó mô phỏng đúng phần cần so sánh cho bottleneck của ta: seed memory kéo thêm memory trong cùng box/neighborhood rồi đưa vào prompt mà không có applicability verifier.

### `aamem`

Method của ta ở Phase 1:

```text
query
  -> retrieve pool
  -> expand evidence neighbors
  -> heuristic applicability gate
  -> authorized packet
```

Gate hiện tại chưa train và không dùng gold evidence. Nó là offline heuristic để test nhanh pipeline, schema, metric và case analysis. Sau này có thể thay bằng LLM judge hoặc classifier mà không đổi format output.

### `oracle`

Upper bound dùng gold ids:

```text
candidate pool -> chỉ giữ gold memory -> packet
```

Không dùng để deploy. Nó cho biết nếu gate hoàn hảo thì trần retrieval+packing cao đến đâu.

## Metric hiện tại

Các metric nằm ở `aamem_lab/metrics.py`:

- `evidence_recall`: trong gold memory, giữ được bao nhiêu.
- `evidence_precision`: memory được đưa vào packet có bao nhiêu là gold.
- `hit_rate`: câu hỏi có ít nhất một gold memory được đưa vào không.
- `noise_rate`: phần memory đưa vào nhưng không phải gold.
- `avg_included`: trung bình số memory vào prompt.
- `avg_tokens`: token estimate của packet/context.

Với claim của ta, không chỉ nhìn recall. Nếu `amem_box` recall tăng nhưng token/noise tăng mạnh, đó chính là khoảng trống cho AAMem.

## Cấu trúc folder

```text
aamem_lab_project/
  aamem_lab/
    schemas.py              # dataclass chuẩn cho memory, query, candidate, gate, result
    datasets.py             # loader ActMemEval và LoCoMo
    retrieval.py            # BM25 + token-neighbor links
    baselines.py            # raw_topk, amem_box, aamem, oracle
    gates.py                # heuristic gate + oracle gate
    packets.py              # raw context và authorized memory packet
    metrics.py              # evidence/token/noise metrics
    geometry.py             # geometry diagnostic: cluster/outlier/drift proxy
    evaluation.py           # chạy method và ghi JSONL/summary
    runner.py               # CLI
    sota_registry.py        # registry so sánh SOTA/proxy hiện tại
    sota_wrappers/
      official_amem.py      # wrapper optional cho repo A-MEM chính thức
  configs/
    phase0_actmem.json
    phase0_locomo.json
  notebooks/
    phase_runner.ipynb
  scripts/
    run_phase0.py
    show_sota_registry.py
  tests/
    smoke_test.py
```

## Chạy từ đầu

Từ folder này:

```powershell
cd G:\Viettel\aamem_lab_project
python -m aamem_lab.runner --dataset actmem --limit 10 --k 5 --out-dir runs/smoke
python -m aamem_lab.runner --dataset locomo --limit 1 --limit-questions 5 --k 5 --out-dir runs/smoke
```

Nếu muốn chạy test:

```powershell
python -m pytest tests/smoke_test.py -q
```

Nếu không có pytest, vẫn có thể chạy CLI trực tiếp vì core code không cần package ngoài.

## Output

Mỗi run sinh:

- `*.jsonl`: từng case, từng method, context/packet, candidate, gate decision, metric.
- `*_summary.json`: aggregate metric theo method.

Một dòng JSONL có dạng:

```json
{
  "case": {
    "id": "...",
    "query": "...",
    "answer": "...",
    "gold_memory_ids": ["..."]
  },
  "result": {
    "method": "aamem_candidate10_packet1200",
    "included_memory_ids": ["..."],
    "context_text": "...",
    "gate_decisions": [...]
  },
  "metrics": {
    "evidence_recall": 0.5,
    "evidence_precision": 0.25,
    "noise_rate": 0.75
  }
}
```

## So sánh với SOTA hiện tại

Phase này so sánh ở đúng tầng retrieval/packing:

- `raw_topk`: baseline retriever.
- `amem_box`: proxy cho A-MEM-style top-k + linked memory expansion.
- `aamem`: method của ta.
- `oracle`: trần nếu biết memory nào áp dụng được.

Không nên claim đã beat full SOTA end-to-end nếu chưa chạy official repo với cùng answerer LLM. Vì vậy repo có `sota_wrappers/official_amem.py` làm wrapper optional. Khi dependency của A-MEM đầy đủ, ta có thể nối nó vào cùng schema `MethodResult`.

Kiểm tra registry:

```powershell
python .\scripts\show_sota_registry.py
```

## Lộ trình phase

### Phase 0: kiểm tra retrieval và token/noise

Mục tiêu:

- loader đúng;
- metric đúng;
- thấy khác biệt giữa `raw_topk`, `amem_box`, `aamem`, `oracle`;
- chọn dataset nào thực sự đánh vào bottleneck.

Dataset:

- ActMemEval subset 10-50 case.
- LoCoMo 1-2 sample, 10-30 question.

### Phase 1: phân tích gate

Mục tiêu:

- đọc từng `gate_decisions`;
- xem memory bị loại có thật sự nhiễu không;
- xem `SUPPORT_ONLY` và `WARNING` có làm packet dễ đọc hơn không;
- chỉnh threshold.

Output cần xem:

- `semantic_relevance`;
- `condition_match`;
- `contradiction_risk`;
- `token_value`;
- `applicability`.

### Phase 2: bọc official A-MEM

Mục tiêu:

- nếu dependency cho phép, chạy official A-MEM trên cùng case;
- so prompt/context của official A-MEM với `amem_box`;
- giữ cùng metric và cùng output schema.

Hiện tại wrapper optional chỉ import an toàn, không ép cài dependency.

### Phase 3: geometry/manifold diagnostic

Mục tiêu:

- không dùng geometry để quyết định truth;
- dùng geometry để phát hiện cluster, outlier, drift, memory box quá dày;
- đưa geometry signal vào gate như feature phụ.

Code hiện có `geometry.py` dùng deterministic token-vector clustering để chạy được ngay. Sau này thay embedding thật cũng không đổi schema.

### Phase 4: stale/continual learning

Mục tiêu:

- test memory update, stale memory, contradiction;
- thêm dataset hoặc synthetic case có memory cũ/mới;
- đánh giá gate có chặn memory cũ không.

Đây là nơi nối STALE/DeltaMem-style idea.

## Notebook

Mở:

```text
notebooks/phase_runner.ipynb
```

Notebook có các cell:

1. setup path;
2. chạy ActMem subset;
3. inspect một case cụ thể theo method;
4. chạy LoCoMo subset;
5. kiểm tra optional official A-MEM wrapper.

## Giới hạn hiện tại

- Đã có answer-level runner, nhưng local mới chạy `heuristic`; muốn đo answer thật thì chạy `--provider openai` hoặc `--provider hf` trên Colab.
- `amem_box` là proxy retrieval/packing, chưa phải full official A-MEM.
- Gate là heuristic chưa train.
- Geometry hiện là diagnostic nhẹ, chưa phải embedding manifold thật.
- Metric gold evidence phụ thuộc annotation; memory hữu ích nhưng không nằm trong gold có thể bị tính là noise.

Nhưng đây là khung đủ để test nhanh claim cốt lõi trên subset thật, rồi nâng dần lên official method, LLM answerer, và dataset lớn hơn.

## Cập nhật: Phase 2 và Phase 3 đã có code

Phần trên là lộ trình ban đầu. Hiện tại folder đã có thêm hai runner mới:

```text
Phase 2: answer-level evaluation
Phase 3: stale-memory stress test
```

### Phase 2: answer-level evaluation

Mục tiêu:

- không chỉ đo memory có sạch không;
- đo xem packet đó giúp model trả lời tốt hơn không;
- log prompt, context, answer, gold answer, evidence metric và answer metric.

Runner:

```powershell
python -m aamem_lab.phase2_answer_runner --dataset actmem --limit 10 --k 5 --provider heuristic --out-dir runs/phase2_answer
```

Với LoCoMo:

```powershell
python -m aamem_lab.phase2_answer_runner --dataset locomo --limit 1 --limit-questions 10 --k 5 --provider heuristic --out-dir runs/phase2_answer
```

Provider model:

- `heuristic`: chạy offline cực nhanh, chỉ để test pipeline và log.
- `openai`: gọi OpenAI-compatible API bằng `OPENAI_API_KEY` hoặc `AAMEM_OPENAI_API_KEY`.
- `hf`: dùng HuggingFace `transformers` trên Colab/GPU.
- Có preset nhỏ trong `aamem_lab/model_presets.py`: `qwen2_5_1_5b`, `qwen2_5_3b`, `qwen2_5_7b`, `llama3_2_1b`, `llama3_2_3b`, `gemma2_2b`.

Xem preset:

```powershell
python .\scripts\show_model_presets.py
```

Vì sao chọn các model nhỏ này:

- A-MEM local/official code trong workspace dùng embedding `all-MiniLM-L6-v2`, mặc định/eval có `gpt-4o-mini`, và hỗ trợ Ollama `llama2/llama3`. Vì vậy small local instruct model là hướng hợp lý để thay API khi test.
- STALE local README có ví dụ chạy target model qua OpenAI-compatible local server với `Qwen3.5-9B`; bản phân tích paper của ta ghi GPT-4o-mini là một backbone chính. Vì chưa có tiền API, ta test trước bằng Qwen/Llama/Gemma 1B-8B.
- DeltaMem local code dùng `deepseek-v4-flash`, không phù hợp với mục tiêu low-cost local run ban đầu.
- LightMem local README cho thấy họ hỗ trợ nhiều backend model và có các setup dùng `gpt-4o-mini`, `qwen3-30b-a3b-instruct`; ta chưa cần chạy cỡ đó ở Phase 2/3.

Preset khuyến nghị chạy theo thứ tự:

```text
smoke -> qwen2_5_1_5b -> qwen2_5_3b -> qwen2_5_7b
```

Nếu HuggingFace account có quyền Llama/Gemma:

```text
llama3_2_1b -> llama3_2_3b
gemma2_2b
```

Judge provider:

- `--judge-provider heuristic`: smoke test local, không phải paper-grade.
- `--judge-provider openai`: model judge kiểu paper hơn, dùng để ra `qa_accuracy`.
- `--judge-provider none`: chỉ log answer, không chấm QA accuracy.

Ví dụ chạy có model thật trên Colab/API:

```powershell
python -m aamem_lab.phase2_answer_runner --dataset actmem --limit 10 --k 5 --provider openai --judge-provider openai --include-prompts --out-dir runs/phase2_answer
```

Metric Phase 2:

- evidence recall;
- evidence precision;
- noise rate;
- context tokens;
- answer exact match;
- answer contains gold;
- answer token F1;
- answer ROUGE-L F1;
- prompt tokens;
- latency.

Metric paper-aligned sinh thêm trong `paper_summary`:

- `retrieval_accuracy`: ActMem-style evidence recall.
- `retrieval_precision`: evidence precision.
- `qa_accuracy`: answer correctness từ judge.
- `qa_score`: soft score từ judge.
- `answer_f1`, `answer_rouge_l`.
- `token_cost`.

Kết quả smoke `ActMemEval n=3, k=3, heuristic`:

```text
method                          ev_recall  ev_prec  noise  ctx_tok  ans_f1
raw_topk_k3                     0.667      0.444    0.556  2001     0.065
amem_style_box_seed3_nbr3       1.000      0.195    0.805  4000     0.065
aamem_candidate10_packet1200    0.833      0.833    0.167  331      0.020
oracle_applicability_candidate20 0.667     1.000    0.000  222      0.092
```

Không đọc quá mạnh vào answer score này vì `heuristic` không phải LLM. Điểm quan trọng là runner đã đo được answer-level. Lên Colab đổi sang `--provider openai` hoặc `--provider hf`.

### Phase 3: stale-memory stress test

Đây là dataset riêng của ta, không phải official STALE dataset. Nó tạo các case có:

- memory cũ nhưng lexical overlap rất cao;
- memory mới/fresh có câu kiểu “changed”, “instead”, “replaced”;
- distractor có cả stale answer và fresh answer;
- ba probe dimension theo tinh thần STALE:
  - `SR`: status recognition, nhận ra memory cũ không còn hợp lệ.
  - `PR`: premise resistance, chống lại câu hỏi gài premise cũ.
  - `IPA`: implicit preference/action, trả lời downstream bằng memory mới.

Runner:

```powershell
python -m aamem_lab.phase3_stale_runner --limit 12 --k 5 --dimensions SR,PR,IPA --provider heuristic --out-dir runs/phase3_stale
```

Method mới:

- `aamem_stale_guard`: AAMem của ta có thêm stale guard.

Pipeline của `aamem_stale_guard`:

```text
retrieve candidates
  -> heuristic gate
  -> tìm newer overlapping update memory
  -> mark older overlapping memory as STALE
  -> promote newer update memory as APPLY
  -> render authorized packet
  -> answerer
```

Metric Phase 3:

- `fresh_memory_hit`: có đưa fresh/current memory vào packet không.
- `stale_context_leak`: stale memory có xuất hiện trong context không.
- `stale_premise_leak`: stale memory có bị dùng như premise không.
- `stale_guard_rate`: có nhận ra stale memory như guard/drop không.
- `fresh_answer_hit`: answer có chứa đáp án fresh không.
- `stale_answer_leak`: answer có rơi vào đáp án cũ không.
- `fresh_over_stale_answer`: answer ưu tiên fresh hơn stale không.
- `stale_probe_accuracy`: accuracy theo từng probe SR/PR/IPA.

Metric paper-aligned sinh thêm trong `paper_summary`:

- `sr_accuracy`;
- `pr_accuracy`;
- `ipa_accuracy`;
- `stale_overall_accuracy`;
- `*_stale_answer_leak`.

Kết quả smoke `synthetic_stale 3 scenarios x SR/PR/IPA, k=3, heuristic`, đọc trong `paper_summary`:

```text
method                              SR_acc  PR_acc  IPA_acc  overall  stale_leak_SR/PR/IPA
raw_topk_k3                         0.667   0.000   0.667    0.444    0.333 / 0.333 / 0.333
amem_style_box_seed3_nbr3           0.667   0.000   0.667    0.444    0.333 / 0.333 / 0.333
aamem_candidate10_packet1200        0.333   0.000   0.333    0.222    0.667 / 0.667 / 0.667
aamem_stale_guard_candidate14...    1.000   1.000   1.000    1.000    0.000 / 0.000 / 0.000
oracle_applicability_candidate20    1.000   1.000   1.000    1.000    0.000 / 0.000 / 0.000
```

Đây là metric mạnh hơn cho claim riêng của ta: A-MEM-style box giữ fresh memory nhưng cũng giữ stale memory như premise; `aamem_stale_guard` loại stale premise và giữ fresh answer trong synthetic stale suite.

### So với hướng SOTA 2026

Registry hiện nằm ở:

```powershell
python .\scripts\show_sota_registry.py
```

Các mốc 2026 đang map vào lab:

- ActMemEval: dataset thật cho evidence/session memory.
- STALE: hướng stale/invalidated memory; Phase 3 đang có synthetic proxy, official loader là bước sau.
- DeltaMem: hướng update/compression; chưa phải bottleneck chính của Phase 2/3.
- MemoryArena / MemoryAgentBench: benchmark tương tác dài; để sau khi answerer ổn.

Vì vậy hiện tại ta đã có:

```text
So với A-MEM 2025: amem_box proxy retrieval/packing.
So với 2026 stale direction: synthetic_stale + aamem_stale_guard metric.
So với 2026 ActMem: chạy trực tiếp ActMemEval subset/full.
```

Chưa claim beat full SOTA 2026 vì chưa chạy official STALE/MemoryArena/MemoryAgentBench protocol.
