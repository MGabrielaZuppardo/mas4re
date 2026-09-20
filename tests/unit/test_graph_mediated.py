"""Unit tests for Condition B: mediated pipeline coordination (ADR-009).

The LLM is fully mocked (deterministic, no Ollama needed), so these tests
verify the graph's conditional routing and retry semantics -- not model
quality. Two requirements are used throughout: "req-01" never conflicts,
"req-02" is a security (SE) requirement whose priority controls whether a
conflict fires (SE + Could/Won't-Have = conflict, per
evaluation/cross_agent_check.py::detect_inter_agent_conflict).
"""

from __future__ import annotations

import json
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from domain.models import PipelineState, Requirement
from pipeline.graph import build_mediated_pipeline_graph


def _requirements() -> list[Requirement]:
    return [
        Requirement(id="req-01", text="The system must generate monthly usage reports."),
        Requirement(id="req-02", text="The system must encrypt all data at rest."),
    ]


def _user_content(messages: list[dict]) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")


def _req_id_from_content(content: str) -> str:
    return "req-02" if "encrypt" in content else "req-01"


def _classify_llm() -> MagicMock:
    """req-01 is always functional; req-02 is always a security NFR."""

    def _side_effect(messages: list[dict]) -> MagicMock:
        req_id = _req_id_from_content(_user_content(messages))
        if req_id == "req-02":
            payload = {
                "requirement_type": "NF",
                "nfr_category": "SE",
                "confidence": 0.9,
                "justification": "security requirement",
            }
        else:
            payload = {
                "requirement_type": "F",
                "nfr_category": None,
                "confidence": 0.9,
                "justification": "functional requirement",
            }
        return MagicMock(content=json.dumps(payload), tool_calls=[])

    llm = MagicMock()
    llm.invoke.side_effect = _side_effect
    llm.bind_tools.return_value = llm
    return llm


def _prioritize_llm(req02_priority_by_occurrence: list[str]) -> MagicMock:
    """req-01 always gets a stable priority; req-02's priority is taken from
    req02_priority_by_occurrence, indexed by how many times req-02 has been
    seen before (occurrence 0 = first pass, occurrence 1 = retry pass)."""

    occurrences: dict[str, int] = defaultdict(int)
    scores = {"M": 1.0, "S": 0.75, "C": 0.5, "W": 0.25}

    def _side_effect(messages: list[dict]) -> MagicMock:
        content = _user_content(messages)
        req_id = _req_id_from_content(content)
        if req_id == "req-01":
            priority = "S"
        else:
            idx = min(occurrences["req-02"], len(req02_priority_by_occurrence) - 1)
            priority = req02_priority_by_occurrence[idx]
            occurrences["req-02"] += 1
        payload = {
            "priority": priority,
            "priority_score": scores[priority],
            "priority_rank": 1,
            "justification": "reasoning",
        }
        return MagicMock(content=json.dumps(payload))

    llm = MagicMock()
    llm.invoke.side_effect = _side_effect
    return llm


def _build_graph(classify_llm: MagicMock, prioritize_llm: MagicMock):
    with (
        patch("agents.classifier.build_llm", return_value=classify_llm),
        patch("agents.prioritizer.build_llm", return_value=prioritize_llm),
    ):
        classifier = ClassificationAgent(model="ollama/qwen2.5:7b")
        prioritizer = PrioritizationAgent(model="ollama/llama3.1:8b")
    return build_mediated_pipeline_graph(classifier, prioritizer)


def test_no_conflict_stops_after_one_pass() -> None:
    """req-02 gets Must-Have -- never conflicts with its SE category, so the
    graph should never take the retry edge."""
    classify_llm = _classify_llm()
    prioritize_llm = _prioritize_llm(["M"])
    graph = _build_graph(classify_llm, prioritize_llm)

    result = graph.invoke(PipelineState(raw_requirements=_requirements()))
    state = PipelineState.model_validate(result) if isinstance(result, dict) else result

    assert state.retry_counts["_pass"] == 1
    assert state.metrics["mediation"] == {
        "conflicts_detected": 0,
        "resent": 0,
        "resolved_after_retry": 0,
        "passes": 1,
    }
    # Classifier: 2 items x primary generation only -- no fault signal
    # (confidence 0.9, type/category consistent), so no self-critique.
    # Prioritizer: 2 items x (primary + self-critique, ADR-011) = 4.
    assert classify_llm.invoke.call_count == 2
    assert prioritize_llm.invoke.call_count == 4


def test_conflict_resolved_after_retry() -> None:
    """req-02 gets Could-Have on pass 1 (conflict: SE + Could-Have), then
    Must-Have on pass 2 -- the retry should resolve it and the graph should
    stop instead of retrying a second time."""
    classify_llm = _classify_llm()
    prioritize_llm = _prioritize_llm(["C", "M"])
    graph = _build_graph(classify_llm, prioritize_llm)

    result = graph.invoke(PipelineState(raw_requirements=_requirements()))
    state = PipelineState.model_validate(result) if isinstance(result, dict) else result

    assert state.retry_counts["_pass"] == 2
    assert state.metrics["mediation"] == {
        "conflicts_detected": 1,
        "resent": 1,
        "resolved_after_retry": 1,
        "passes": 2,
    }
    # Whole batch resent -> both items reclassified/reprioritized on pass 2.
    # Classifier: 2 items x primary only (no fault signal) x 2 passes = 4.
    # Prioritizer: 2 items x (primary + self-critique) x 2 passes = 8.
    assert classify_llm.invoke.call_count == 4
    assert prioritize_llm.invoke.call_count == 8
    assert state.prioritized_requirements[0].priority_rank is not None


def test_conflict_persists_stops_at_max_passes() -> None:
    """req-02 gets Could-Have on every pass -- the conflict never resolves,
    but the graph must still terminate after 2 passes (no infinite loop)."""
    classify_llm = _classify_llm()
    prioritize_llm = _prioritize_llm(["C"])
    graph = _build_graph(classify_llm, prioritize_llm)

    result = graph.invoke(PipelineState(raw_requirements=_requirements()))
    state = PipelineState.model_validate(result) if isinstance(result, dict) else result

    assert state.retry_counts["_pass"] == 2
    assert state.metrics["mediation"] == {
        "conflicts_detected": 1,
        "resent": 1,
        "resolved_after_retry": 0,
        "passes": 2,
    }


@pytest.mark.parametrize("fixture_priorities", [["M"], ["C", "M"], ["C"]])
def test_condition_a_graph_unaffected(fixture_priorities: list[str]) -> None:
    """Sanity check that Condition A (build_pipeline_graph) never reads the
    mediation fields cross_check_node now writes -- same node, different
    graph topology, no behavior change for the fixed-flow condition."""
    from pipeline.graph import build_pipeline_graph

    classify_llm = _classify_llm()
    prioritize_llm = _prioritize_llm(fixture_priorities)

    with (
        patch("agents.classifier.build_llm", return_value=classify_llm),
        patch("agents.prioritizer.build_llm", return_value=prioritize_llm),
    ):
        classifier = ClassificationAgent(model="ollama/qwen2.5:7b")
        prioritizer = PrioritizationAgent(model="ollama/llama3.1:8b")
    graph = build_pipeline_graph(classifier, prioritizer)

    result = graph.invoke(PipelineState(raw_requirements=_requirements()))
    state = PipelineState.model_validate(result) if isinstance(result, dict) else result

    # cross_check_node still runs once and sets these, but nothing routes on
    # them -- Condition A is always exactly one pass regardless of conflict.
    assert state.retry_counts["_pass"] == 1
    # Classifier: 2 items x primary generation only -- no fault signal
    # (confidence 0.9, type/category consistent), so no self-critique.
    # Prioritizer: 2 items x (primary + self-critique, ADR-011) = 4.
    assert classify_llm.invoke.call_count == 2
    assert prioritize_llm.invoke.call_count == 4
