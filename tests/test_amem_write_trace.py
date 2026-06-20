from __future__ import annotations

from aamem_lab.amem_write_trace_runner import (
    EVOLVE_DECISION,
    LINK_STRENGTHEN,
    NEIGHBOR_UPDATE,
    NOTE_ANALYSIS,
    _connection_diagnostics,
    _format_quality,
    classify_amem_write_prompt,
)


def test_classify_official_write_prompts() -> None:
    assert classify_amem_write_prompt("Analyze the following content and provide:\n1. KEYWORDS") == NOTE_ANALYSIS
    assert classify_amem_write_prompt("You are an AI memory evolution agent. Analyze the new memory") == EVOLVE_DECISION
    assert (
        classify_amem_write_prompt(
            "Given the new memory and its neighbors, provide updated connections and tags."
        )
        == LINK_STRENGTHEN
    )
    assert (
        classify_amem_write_prompt(
            "Given the new memory and its neighbor memories, update each neighbor's context and tags"
        )
        == NEIGHBOR_UPDATE
    )


def test_connection_diagnostics_separates_invalid_and_non_neighbor_links() -> None:
    calls = [
        {
            "kind": LINK_STRENGTHEN,
            "parsed_output": {"connections": [1, 3, 8, 3]},
            "retrieved_neighbor_indices": [1, 3, 4],
        }
    ]
    result = _connection_diagnostics(calls, memory_count_before=5)
    assert result["valid_memory_indices"] == [1, 3, 3]
    assert result["invalid_memory_indices"] == [8]
    assert result["not_in_retrieved_neighbor_set"] == [8]
    assert result["duplicate_links"] == [3]


def test_arbitrary_json_is_not_counted_as_valid_note_format() -> None:
    quality = _format_quality(
        NOTE_ANALYSIS,
        '{"foo": "bar"}',
        {"keywords": ["repaired"], "context": "repaired", "tags": ["repaired"]},
        "Analyze the following content and provide:",
    )
    assert quality["raw_is_json_object"] is True
    assert quality["json_schema_ok"] is False
    assert quality["format_ok"] is False
