"""Unit tests for MediatedPipelineStrategy (Condition B, ADR-009).

Mirrors the level of coverage PipelineStrategy gets implicitly via
test_pipeline_streaming.py: construction, name, delegation to the compiled
graph, and trace_writer propagation to both agents.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from domain.models import Requirement
from experiments.strategy import MediatedPipelineStrategy


def _requirements() -> list[Requirement]:
    return [Requirement(id="req-01", text="The system must respond within 200ms.")]


def _classify_llm() -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content=json.dumps(
            {
                "requirement_type": "NF",
                "nfr_category": "PE",
                "confidence": 0.9,
                "justification": "performance requirement",
            }
        ),
        tool_calls=[],
    )
    llm.bind_tools.return_value = llm
    return llm


def _prioritize_llm() -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content=json.dumps(
            {
                "priority": "M",
                "priority_score": 1.0,
                "priority_rank": 1,
                "justification": "critical for the system",
            }
        )
    )
    return llm


def test_name_is_pipeline_mediated() -> None:
    with (
        patch("agents.classifier.build_llm", return_value=_classify_llm()),
        patch("agents.prioritizer.build_llm", return_value=_prioritize_llm()),
    ):
        strategy = MediatedPipelineStrategy(
            classifier_model="ollama/qwen2.5:7b", prioritizer_model="ollama/llama3.1:8b"
        )
    assert strategy.name == "pipeline_mediated"


def test_execute_returns_prioritized_requirements() -> None:
    classify_llm = _classify_llm()
    prioritize_llm = _prioritize_llm()
    with (
        patch("agents.classifier.build_llm", return_value=classify_llm),
        patch("agents.prioritizer.build_llm", return_value=prioritize_llm),
    ):
        strategy = MediatedPipelineStrategy(
            classifier_model="ollama/qwen2.5:7b", prioritizer_model="ollama/llama3.1:8b"
        )
        state = strategy.execute(_requirements())

    assert len(state.prioritized_requirements) == 1
    assert state.prioritized_requirements[0].priority.value == "M"
    # No conflict (PE + Must-Have) -> single pass, no retry.
    assert state.retry_counts["_pass"] == 1
    assert state.metrics["mediation"]["resent"] == 0


def test_execute_propagates_trace_writer() -> None:
    classify_llm = _classify_llm()
    prioritize_llm = _prioritize_llm()
    trace_writer = MagicMock()
    with (
        patch("agents.classifier.build_llm", return_value=classify_llm),
        patch("agents.prioritizer.build_llm", return_value=prioritize_llm),
    ):
        strategy = MediatedPipelineStrategy(
            classifier_model="ollama/qwen2.5:7b", prioritizer_model="ollama/llama3.1:8b"
        )
        strategy.execute(_requirements(), trace_writer=trace_writer)

    assert strategy._classifier._trace is trace_writer
    assert strategy._prioritizer._trace is trace_writer
    assert trace_writer.write.call_count == 2  # one classify call + one prioritize call
