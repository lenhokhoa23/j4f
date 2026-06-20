from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .official_amem_gate_runner import HFCompatibleRobustLLMController


NOTE_ANALYSIS = "note_analysis"
KEYWORD_RETRY = "keyword_retry"
EVOLVE_DECISION = "evolve_decision"
LINK_STRENGTHEN = "link_strengthen"
NEIGHBOR_UPDATE = "neighbor_update"
UNKNOWN_CALL = "unknown"

VALID_EVOLUTION_DECISIONS = {
    "NO_EVOLUTION",
    "STRENGTHEN",
    "UPDATE_NEIGHBOR",
    "STRENGTHEN_AND_UPDATE",
}

PLACEHOLDER_PATTERNS = (
    r"\bupdated context sentence\b",
    r"\bkeyword\d+\b",
    r"\btag\d+\b",
    r"\bneighbor_memory_ids\b",
    r"\bbrief explanation\b",
    r"[\"']none[\"']",
    r"[\"']n/?a[\"']",
)


def classify_amem_write_prompt(prompt: str) -> str:
    """Identify which official robust A-MEM write-time call produced a prompt."""

    text = prompt.lstrip()
    if text.startswith("Analyze the following content and provide:"):
        return NOTE_ANALYSIS
    if text.startswith("List exactly 5 keywords that capture the main concepts"):
        return KEYWORD_RETRY
    if text.startswith("You are an AI memory evolution agent"):
        return EVOLVE_DECISION
    if text.startswith("Given the new memory and its neighbors, provide updated connections and tags"):
        return LINK_STRENGTHEN
    if text.startswith("Given the new memory and its neighbor memories, update each neighbor's context and tags"):
        return NEIGHBOR_UPDATE
    return UNKNOWN_CALL


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _preview(value: Any, max_chars: int = 180) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _contains_placeholder(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, default=_json_default) if not isinstance(value, str) else value
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PLACEHOLDER_PATTERNS)


def _extract_neighbor_indices(prompt: str) -> List[int]:
    return [int(value) for value in re.findall(r"memory index:\s*(\d+)", prompt, flags=re.IGNORECASE)]


def _extract_neighbor_count(prompt: str) -> int:
    match = re.search(r"continue for all\s+(\d+)\s+neighbors", prompt, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    indices = _extract_neighbor_indices(prompt)
    return len(indices)


def _raw_declared_decision(raw: str) -> str:
    match = re.search(r"^\s*DECISION\s*:\s*([^\n\r]+)", raw, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    return match.group(1).strip().upper().replace(" ", "_")


@dataclass
class RecordingLLM:
    """Transparent wrapper that records official A-MEM LLM prompts and outputs."""

    inner: Any

    def __post_init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        started = time.perf_counter()
        call: Dict[str, Any] = {
            "call_index": len(self.calls),
            "kind": classify_amem_write_prompt(prompt),
            "temperature": float(temperature),
            "prompt": prompt,
            "raw_output": "",
            "error": "",
        }
        try:
            output = self.inner.get_completion(prompt, temperature=temperature)
            call["raw_output"] = str(output or "")
            return output
        except Exception as exc:
            call["error"] = f"{exc.__class__.__name__}: {exc}"
            raise
        finally:
            call["elapsed_sec"] = time.perf_counter() - started
            self.calls.append(call)


def _load_write_modules(amem_repo: Path) -> Dict[str, Any]:
    repo = amem_repo.resolve()
    required = ("memory_layer_robust.py", "llm_text_parsers.py", "load_dataset.py")
    missing = [name for name in required if not (repo / name).exists()]
    if missing:
        raise FileNotFoundError(f"Not an official robust A-MEM root: {repo}; missing={missing}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return {
        "memory_layer": importlib.import_module("memory_layer_robust"),
        "parsers": importlib.import_module("llm_text_parsers"),
        "dataset": importlib.import_module("load_dataset"),
    }


def _patch_hf_backend(memory_layer: Any, max_new_tokens: int) -> None:
    original_cls = getattr(memory_layer, "_aamem_write_trace_original_controller", None)
    if original_cls is None:
        original_cls = memory_layer.RobustLLMController
        memory_layer._aamem_write_trace_original_controller = original_cls

    class PatchedRobustLLMController(HFCompatibleRobustLLMController):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(
                *args,
                original_cls=original_cls,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

    memory_layer.RobustLLMController = PatchedRobustLLMController


def _snapshot_note(note: Any, index: int) -> Dict[str, Any]:
    links = getattr(note, "links", []) or []
    return {
        "index": int(index),
        "id": str(getattr(note, "id", "")),
        "timestamp": str(getattr(note, "timestamp", "")),
        "content": str(getattr(note, "content", "")),
        "context": str(getattr(note, "context", "")),
        "keywords": list(getattr(note, "keywords", []) or []),
        "tags": list(getattr(note, "tags", []) or []),
        "links": [int(value) for value in links],
    }


def _snapshot_memory_system(system: Any) -> Dict[str, Dict[str, Any]]:
    return {
        str(note_id): _snapshot_note(note, index)
        for index, (note_id, note) in enumerate(system.memories.items())
    }


def _diff_existing_memories(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []
    mutable_fields = ("context", "keywords", "tags", "links")
    for note_id, old in before.items():
        new = after.get(note_id)
        if new is None:
            changes.append({"id": note_id, "index": old["index"], "deleted": True})
            continue
        field_changes = {
            field: {"before": old[field], "after": new[field]}
            for field in mutable_fields
            if old[field] != new[field]
        }
        if field_changes:
            changes.append(
                {
                    "id": note_id,
                    "index": old["index"],
                    "content": old["content"],
                    "fields": field_changes,
                }
            )
    return changes


def _format_quality(kind: str, raw: str, parsed: Any, prompt: str) -> Dict[str, Any]:
    json_object = _parse_json_object(raw)
    json_keys = set(json_object or {})
    expected_json_keys: Sequence[set[str]]
    markers: Sequence[str]
    if kind == NOTE_ANALYSIS:
        markers = ("KEYWORDS:", "CONTEXT:", "TAGS:")
        expected_json_keys = ({"keywords", "context", "tags"},)
    elif kind == EVOLVE_DECISION:
        markers = ("DECISION:", "REASON:")
        expected_json_keys = ({"decision"}, {"should_evolve", "actions"})
    elif kind == LINK_STRENGTHEN:
        markers = ("CONNECTIONS:", "TAGS:")
        expected_json_keys = ({"connections", "tags"}, {"suggested_connections", "tags_to_update"})
    elif kind == NEIGHBOR_UPDATE:
        neighbor_count = _extract_neighbor_count(prompt)
        markers = tuple(f"NEIGHBOR {idx}:" for idx in range(neighbor_count))
        expected_json_keys = ({"new_context_neighborhood", "new_tags_neighborhood"},)
    else:
        markers = ()
        expected_json_keys = ()

    marker_coverage = 1.0
    if markers:
        found = sum(1 for marker in markers if marker.lower() in raw.lower())
        marker_coverage = found / len(markers)

    json_schema_ok = bool(json_object) and any(required.issubset(json_keys) for required in expected_json_keys)
    format_ok = bool(raw.strip()) and (json_schema_ok or marker_coverage == 1.0)
    parser_repair_suspected = False
    if kind == NOTE_ANALYSIS and isinstance(parsed, dict):
        parser_repair_suspected = marker_coverage < 1.0 and any(parsed.get(key) for key in ("keywords", "context", "tags"))
    elif kind == EVOLVE_DECISION and isinstance(parsed, dict):
        declared = _raw_declared_decision(raw)
        parser_repair_suspected = declared not in VALID_EVOLUTION_DECISIONS

    return {
        "raw_is_json_object": json_object is not None,
        "raw_json_keys": sorted(json_keys),
        "json_schema_ok": json_schema_ok,
        "marker_coverage": marker_coverage,
        "format_ok": format_ok,
        "parser_repair_suspected": parser_repair_suspected,
        "placeholder_detected": _contains_placeholder(raw) or _contains_placeholder(parsed),
    }


def parse_recorded_call(call: Dict[str, Any], parsers: Any, turn_content: str) -> Dict[str, Any]:
    """Parse a recorded call with the same parser functions used by robust A-MEM."""

    record = dict(call)
    raw = str(record.get("raw_output", ""))
    kind = str(record.get("kind", UNKNOWN_CALL))
    parsed: Any = None
    parse_error = ""
    try:
        if record.get("error"):
            parsed = None
        elif kind == NOTE_ANALYSIS:
            parsed = parsers.parse_analyze_content(raw, turn_content)
        elif kind == KEYWORD_RETRY:
            parsed = parsers._parse_list_items(raw)
        elif kind == EVOLVE_DECISION:
            parsed = parsers.parse_evolution_decision(raw)
        elif kind == LINK_STRENGTHEN:
            parsed = parsers.parse_strengthen_details(raw)
        elif kind == NEIGHBOR_UPDATE:
            parsed = parsers.parse_update_neighbors(raw, _extract_neighbor_count(record.get("prompt", "")))
        else:
            parsed = raw
    except Exception as exc:
        parse_error = f"{exc.__class__.__name__}: {exc}"

    record["parsed_output"] = parsed
    record["parse_error"] = parse_error
    record["quality"] = _format_quality(kind, raw, parsed, str(record.get("prompt", "")))
    if kind in (EVOLVE_DECISION, LINK_STRENGTHEN, NEIGHBOR_UPDATE):
        record["retrieved_neighbor_indices"] = _extract_neighbor_indices(str(record.get("prompt", "")))
    return record


def _connection_diagnostics(
    calls: Sequence[Dict[str, Any]],
    memory_count_before: int,
) -> Dict[str, Any]:
    proposed: List[int] = []
    retrieved: List[int] = []
    for call in calls:
        if call.get("kind") != LINK_STRENGTHEN:
            continue
        parsed = call.get("parsed_output") or {}
        proposed.extend(int(value) for value in parsed.get("connections", []) if isinstance(value, (int, float)))
        retrieved.extend(int(value) for value in call.get("retrieved_neighbor_indices", []))
    retrieved_set = set(retrieved)
    return {
        "retrieved_neighbor_indices": list(dict.fromkeys(retrieved)),
        "proposed_links": proposed,
        "valid_memory_indices": [value for value in proposed if 0 <= value < memory_count_before],
        "invalid_memory_indices": [value for value in proposed if value < 0 or value >= memory_count_before],
        "not_in_retrieved_neighbor_set": [value for value in proposed if value not in retrieved_set],
        "duplicate_links": sorted(value for value, count in Counter(proposed).items() if count > 1),
    }


def _evolution_diagnostics(calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    decision_calls = [call for call in calls if call.get("kind") == EVOLVE_DECISION]
    link_calls = [call for call in calls if call.get("kind") == LINK_STRENGTHEN]
    update_calls = [call for call in calls if call.get("kind") == NEIGHBOR_UPDATE]
    if not decision_calls:
        decision = "SKIPPED_NO_NEIGHBORS"
    else:
        parsed = decision_calls[-1].get("parsed_output") or {}
        decision = str(parsed.get("decision", "UNKNOWN"))

    expects_link = decision in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
    expects_update = decision in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")
    update_blocks = []
    for call in update_calls:
        parsed = call.get("parsed_output")
        if isinstance(parsed, list):
            update_blocks.extend(parsed)
    nonempty_updates = sum(
        1 for item in update_blocks if isinstance(item, dict) and (item.get("context") or item.get("tags"))
    )
    return {
        "decision": decision,
        "decision_call_count": len(decision_calls),
        "link_call_count": len(link_calls),
        "update_call_count": len(update_calls),
        "expected_link_call": expects_link,
        "expected_update_call": expects_update,
        "conditional_call_match": (expects_link == bool(link_calls)) and (expects_update == bool(update_calls)),
        "parsed_update_block_count": len(update_blocks),
        "nonempty_update_block_count": nonempty_updates,
    }


def _flatten_turns(sample: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for session_id, session in sorted(sample.conversation.sessions.items(), key=lambda item: int(item[0])):
        for turn in session.turns:
            rows.append(
                {
                    "session_id": int(session_id),
                    "dia_id": str(turn.dia_id),
                    "speaker": str(turn.speaker),
                    "text": str(turn.text),
                    "timestamp": str(session.date_time),
                    "amem_content": "Speaker " + str(turn.speaker) + "says : " + str(turn.text),
                }
            )
    return rows


def _trace_turn(
    system: Any,
    recorder: RecordingLLM,
    parsers: Any,
    turn: Dict[str, Any],
    turn_index: int,
    model: str,
    backend: str,
    include_prompts: bool,
) -> Dict[str, Any]:
    before = _snapshot_memory_system(system)
    memory_count_before = len(before)
    call_start = len(recorder.calls)
    started = time.perf_counter()
    note_id = system.add_note(turn["amem_content"], time=turn["timestamp"])
    elapsed = time.perf_counter() - started
    raw_calls = recorder.calls[call_start:]
    calls = [parse_recorded_call(call, parsers, turn["amem_content"]) for call in raw_calls]

    after = _snapshot_memory_system(system)
    new_note = after[str(note_id)]
    existing_changes = _diff_existing_memories(before, after)
    connection = _connection_diagnostics(calls, memory_count_before)
    evolution = _evolution_diagnostics(calls)

    note_calls = [call for call in calls if call.get("kind") == NOTE_ANALYSIS]
    note_quality = note_calls[-1]["quality"] if note_calls else {
        "format_ok": False,
        "parser_repair_suspected": True,
        "placeholder_detected": _contains_placeholder(new_note),
        "marker_coverage": 0.0,
        "raw_is_json_object": False,
    }

    if not include_prompts:
        for call in calls:
            call.pop("prompt", None)

    return {
        "model": model,
        "backend": backend,
        "turn_index": int(turn_index),
        "dia_id": turn["dia_id"],
        "session_id": turn["session_id"],
        "speaker": turn["speaker"],
        "timestamp": turn["timestamp"],
        "turn_text": turn["text"],
        "amem_content": turn["amem_content"],
        "elapsed_sec": elapsed,
        "memory_count_before": memory_count_before,
        "memory_count_after": len(after),
        "new_memory": new_note,
        "note_quality": note_quality,
        "evolution": evolution,
        "connections": connection,
        "existing_memory_changes": existing_changes,
        "calls": calls,
    }


def _iter_call_rows(turn_rows: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for turn in turn_rows:
        for call in turn["calls"]:
            yield {
                "model": turn["model"],
                "backend": turn["backend"],
                "turn_index": turn["turn_index"],
                "dia_id": turn["dia_id"],
                "memory_count_before": turn["memory_count_before"],
                "kind": call["kind"],
                "temperature": call["temperature"],
                "elapsed_sec": call["elapsed_sec"],
                "error": call["error"],
                "parse_error": call["parse_error"],
                "quality": call["quality"],
                "parsed_output": call["parsed_output"],
                "raw_output": call["raw_output"],
                "prompt": call.get("prompt", ""),
            }


def summarize_trace(turn_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    stage_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for call in _iter_call_rows(turn_rows):
        stage_rows[call["kind"]].append(call)

    stages: Dict[str, Any] = {}
    for kind, calls in stage_rows.items():
        total = len(calls)
        stages[kind] = {
            "calls": total,
            "format_ok": sum(bool(call["quality"].get("format_ok")) for call in calls),
            "format_ok_rate": sum(bool(call["quality"].get("format_ok")) for call in calls) / total,
            "parser_repair_suspected": sum(
                bool(call["quality"].get("parser_repair_suspected")) for call in calls
            ),
            "placeholder_detected": sum(
                bool(call["quality"].get("placeholder_detected")) for call in calls
            ),
            "llm_errors": sum(bool(call.get("error")) for call in calls),
            "parse_errors": sum(bool(call.get("parse_error")) for call in calls),
            "mean_elapsed_sec": sum(float(call["elapsed_sec"]) for call in calls) / total,
        }

    decisions = Counter(row["evolution"]["decision"] for row in turn_rows)
    proposed_links = sum(len(row["connections"]["proposed_links"]) for row in turn_rows)
    invalid_links = sum(len(row["connections"]["invalid_memory_indices"]) for row in turn_rows)
    out_of_neighbor_links = sum(len(row["connections"]["not_in_retrieved_neighbor_set"]) for row in turn_rows)
    changed_neighbors = sum(len(row["existing_memory_changes"]) for row in turn_rows)
    return {
        "model": turn_rows[0]["model"] if turn_rows else "",
        "backend": turn_rows[0]["backend"] if turn_rows else "",
        "turns": len(turn_rows),
        "total_llm_calls": sum(len(row["calls"]) for row in turn_rows),
        "mean_turn_elapsed_sec": (
            sum(float(row["elapsed_sec"]) for row in turn_rows) / len(turn_rows) if turn_rows else 0.0
        ),
        "stages": stages,
        "evolution_decisions": dict(decisions),
        "proposed_links": proposed_links,
        "invalid_memory_index_links": invalid_links,
        "links_outside_retrieved_neighbor_set": out_of_neighbor_links,
        "existing_memories_changed": changed_neighbors,
    }


def _print_turn_trace(
    row: Dict[str, Any],
    print_prompts: bool,
    print_raw_outputs: bool,
    max_output_chars: int,
) -> None:
    evo = row["evolution"]
    links = row["connections"]
    print(
        f"\n[turn] idx={row['turn_index']} dia_id={row['dia_id']} speaker={row['speaker']} "
        f"memories={row['memory_count_before']}->{row['memory_count_after']} "
        f"calls={len(row['calls'])} decision={evo['decision']} elapsed={row['elapsed_sec']:.2f}s",
        flush=True,
    )
    print(f"[turn text] {row['turn_text']}", flush=True)
    note = row["new_memory"]
    print(
        f"[note] keywords={note['keywords']} context={note['context']} tags={note['tags']} links={note['links']}",
        flush=True,
    )
    print(
        f"[link] retrieved={links['retrieved_neighbor_indices']} proposed={links['proposed_links']} "
        f"invalid={links['invalid_memory_indices']} outside_retrieved={links['not_in_retrieved_neighbor_set']}",
        flush=True,
    )
    if row["existing_memory_changes"]:
        print(f"[evolve diff] changed_existing={len(row['existing_memory_changes'])}", flush=True)
        for change in row["existing_memory_changes"]:
            print(
                f"  memory_index={change['index']} fields={list(change.get('fields', {}))} "
                f"content={_preview(change.get('content', ''))}",
                flush=True,
            )
    else:
        print("[evolve diff] changed_existing=0", flush=True)

    for call in row["calls"]:
        quality = call["quality"]
        print(
            f"[call {call['call_index']}] kind={call['kind']} elapsed={call['elapsed_sec']:.2f}s "
            f"format_ok={quality['format_ok']} repair={quality['parser_repair_suspected']} "
            f"placeholder={quality['placeholder_detected']} error={call['error'] or '-'}",
            flush=True,
        )
        if print_prompts and call.get("prompt"):
            prompt = str(call["prompt"])
            print("--- PROMPT ---", flush=True)
            print(prompt[:max_output_chars], flush=True)
        if print_raw_outputs:
            raw = str(call["raw_output"])
            print("--- RAW OUTPUT ---", flush=True)
            print(raw[:max_output_chars], flush=True)
        print("--- PARSED OUTPUT ---", flush=True)
        print(json.dumps(call["parsed_output"], ensure_ascii=False, indent=2, default=_json_default), flush=True)


def run_trace(args: argparse.Namespace) -> Dict[str, Any]:
    _seed_everything(args.seed)
    modules = _load_write_modules(Path(args.amem_repo))
    if args.backend == "hf":
        _patch_hf_backend(modules["memory_layer"], args.hf_max_new_tokens)

    samples = modules["dataset"].load_locomo_dataset(Path(args.dataset))
    if args.sample_idx < 0 or args.sample_idx >= len(samples):
        raise IndexError(f"sample_idx={args.sample_idx} outside dataset with {len(samples)} samples")
    all_turns = _flatten_turns(samples[args.sample_idx])
    selected_turns = all_turns[: args.max_turns] if args.max_turns > 0 else all_turns
    if not selected_turns:
        raise ValueError("No LoCoMo turns selected for tracing")

    system = modules["memory_layer"].RobustAgenticMemorySystem(
        model_name=args.embedding_model,
        llm_backend=args.backend,
        llm_model=args.model,
        evo_threshold=args.evo_threshold,
        sglang_host=args.host,
        sglang_port=args.port,
    )
    recorder = RecordingLLM(system.llm_controller.llm)
    system.llm_controller.llm = recorder

    print(
        f"[config] model={args.model} backend={args.backend} sample={args.sample_idx} "
        f"turns={len(selected_turns)}/{len(all_turns)} embedding={args.embedding_model}",
        flush=True,
    )
    rows: List[Dict[str, Any]] = []
    for turn_index, turn in enumerate(selected_turns):
        row = _trace_turn(
            system=system,
            recorder=recorder,
            parsers=modules["parsers"],
            turn=turn,
            turn_index=turn_index,
            model=args.model,
            backend=args.backend,
            include_prompts=args.include_prompts,
        )
        rows.append(row)
        _print_turn_trace(
            row,
            print_prompts=args.print_prompts,
            print_raw_outputs=args.print_raw_outputs,
            max_output_chars=args.max_output_chars,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"write_trace_s{args.sample_idx}_n{len(rows)}_{args.backend}_{_safe_name(args.model)}"
    trace_path = output_dir / f"{tag}.jsonl"
    calls_path = output_dir / f"{tag}_calls.jsonl"
    summary_path = output_dir / f"{tag}_summary.json"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    with calls_path.open("w", encoding="utf-8") as handle:
        for call in _iter_call_rows(rows):
            handle.write(json.dumps(call, ensure_ascii=False, default=_json_default) + "\n")

    summary = summarize_trace(rows)
    summary["config"] = {
        "amem_repo": str(Path(args.amem_repo).resolve()),
        "dataset": str(Path(args.dataset).resolve()),
        "sample_idx": args.sample_idx,
        "max_turns": args.max_turns,
        "model": args.model,
        "backend": args.backend,
        "hf_max_new_tokens": args.hf_max_new_tokens,
        "embedding_model": args.embedding_model,
        "include_prompts": args.include_prompts,
        "seed": args.seed,
    }
    summary["outputs"] = {
        "turn_trace_jsonl": str(trace_path.resolve()),
        "call_trace_jsonl": str(calls_path.resolve()),
        "summary_json": str(summary_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    print("\n=== A-MEM write-time trace summary ===", flush=True)
    print(
        f"model={summary['model']} turns={summary['turns']} calls={summary['total_llm_calls']} "
        f"decisions={summary['evolution_decisions']}",
        flush=True,
    )
    print(
        f"links={summary['proposed_links']} invalid={summary['invalid_memory_index_links']} "
        f"outside_retrieved={summary['links_outside_retrieved_neighbor_set']} "
        f"existing_memories_changed={summary['existing_memories_changed']}",
        flush=True,
    )
    for kind, stats in summary["stages"].items():
        print(
            f"stage={kind:<18} calls={stats['calls']:<3} format_ok={stats['format_ok']:<3} "
            f"repair={stats['parser_repair_suspected']:<3} placeholders={stats['placeholder_detected']:<3} "
            f"errors={stats['llm_errors'] + stats['parse_errors']:<3} mean_sec={stats['mean_elapsed_sec']:.2f}",
            flush=True,
        )
    print(f"Wrote turn traces: {trace_path}", flush=True)
    print(f"Wrote call traces: {calls_path}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace raw note/link/evolve outputs from official robust A-MEM without changing its write logic."
    )
    parser.add_argument("--amem-repo", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backend", default="hf", choices=("hf", "openai", "ollama", "sglang", "vllm"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--hf-max-new-tokens", type=int, default=768)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--evo-threshold", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="http://localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--include-prompts", action="store_true")
    parser.add_argument("--print-prompts", action="store_true")
    parser.add_argument("--print-raw-outputs", action="store_true")
    parser.add_argument("--max-output-chars", type=int, default=12000)
    parser.add_argument("--output-dir", default="runs/amem_write_trace")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    if args.max_turns == 0 or args.max_turns < -1:
        parser.error("--max-turns must be -1 (all) or a positive integer")
    if args.hf_max_new_tokens <= 0:
        parser.error("--hf-max-new-tokens must be positive")
    return args


def main() -> None:
    run_trace(parse_args())


if __name__ == "__main__":
    main()
