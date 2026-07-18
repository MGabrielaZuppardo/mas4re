"""Equivalence tests: PipelineStreamingStrategy vs. PipelineStrategy.

Both must produce the same classified/prioritized content (content compared
after sorting by requirement id, since the streaming graph's branches finish
in completion order, not input order) and the same failure-isolation
behavior for a failing item — the only thing allowed to differ is scheduling.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from domain.models import Requirement
from experiments.strategy import PipelineStrategy, PipelineStreamingStrategy

_N = 8


def _requirement_text(i: int) -> str:
    return f"The system must handle REQ{i:02d} securely and respond quickly."


@pytest.fixture
def requirements() -> list[Requirement]:
    return [Requirement(id=f"req-{i:02d}", text=_requirement_text(i)) for i in range(_N)]


def _user_content(messages: list[dict]) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")


def _index_from_content(content: str) -> int:
    for i in range(_N):
        if f"REQ{i:02d}" in content:
            return i
    raise AssertionError(f"no REQ marker found in {content!r}")


def _classify_side_effect(messages: list[dict]) -> MagicMock:
    i = _index_from_content(_user_content(messages))
    req_type = "NF" if i % 2 == 0 else "F"
    nfr_category = "SE" if req_type == "NF" else None
    confidence = round(0.5 + (i % 5) * 0.1, 2)
    payload = {
        "requirement_type": req_type,
        "nfr_category": nfr_category,
        "confidence": confidence,
        "justification": f"reasoning about REQ{i:02d}",
    }
    return MagicMock(content=json.dumps(payload))


def _prioritize_side_effect(messages: list[dict]) -> MagicMock:
    i = _index_from_content(_user_content(messages))
    # Deliberate ties: items 0/1 and 2/3 share a score, to exercise
    # rank_by_priority's tie-break stability across both strategies.
    score = [1.0, 1.0, 0.75, 0.75, 0.5, 0.25, 0.1, 0.9][i]
    priority = {1.0: "M", 0.75: "S", 0.5: "C", 0.25: "W", 0.1: "W", 0.9: "M"}[score]
    content = (
        f'{{"priority": "{priority}", "priority_score": {score}, "priority_rank": 1,'
        f' "justification": "reasoning about REQ{i:02d}"}}'
    )
    return MagicMock(content=content)


@pytest.fixture
def mocked_llms():
    classify_llm = MagicMock()
    classify_llm.invoke.side_effect = _classify_side_effect
    prioritize_llm = MagicMock()
    prioritize_llm.invoke.side_effect = _prioritize_side_effect
    with (
        patch("agents.classifier.build_llm", return_value=classify_llm),
        patch("agents.prioritizer.build_llm", return_value=prioritize_llm),
    ):
        yield


def _content_by_id(state) -> dict[str, tuple]:
    return {
        p.id: (
            p.requirement_type.value,
            p.nfr_category,
            round(p.confidence, 4),
            p.priority.value,
            round(p.priority_score, 4),
            p.priority_rank,
        )
        for p in state.prioritized_requirements
    }


def test_streaming_matches_batch_pipeline_content(mocked_llms, requirements) -> None:
    batch_state = PipelineStrategy(
        classifier_model="ollama/qwen2.5:7b", prioritizer_model="ollama/llama3.1:8b"
    ).execute(requirements)
    streaming_state = PipelineStreamingStrategy(
        classifier_model="ollama/qwen2.5:7b", prioritizer_model="ollama/llama3.1:8b"
    ).execute(requirements)

    assert len(batch_state.classified_requirements) == _N
    assert len(streaming_state.classified_requirements) == _N
    assert _content_by_id(batch_state) == _content_by_id(streaming_state)


def test_streaming_failure_isolation_matches_batch(mocked_llms, requirements) -> None:
    """A single failing item must be dropped from both strategies' output
    identically -- one bad item doesn't take any other item down with it,
    and doesn't change the survivors' content."""

    def flaky_classify(messages: list[dict]) -> MagicMock:
        if _index_from_content(_user_content(messages)) == 3:
            raise ConnectionError("simulated transient failure")
        return _classify_side_effect(messages)

    classify_llm = MagicMock()
    classify_llm.invoke.side_effect = flaky_classify
    prioritize_llm = MagicMock()
    prioritize_llm.invoke.side_effect = _prioritize_side_effect

    with (
        patch("agents.classifier.build_llm", return_value=classify_llm),
        patch("agents.prioritizer.build_llm", return_value=prioritize_llm),
    ):
        batch_state = PipelineStrategy(
            classifier_model="ollama/qwen2.5:7b", prioritizer_model="ollama/llama3.1:8b"
        ).execute(requirements)
        streaming_state = PipelineStreamingStrategy(
            classifier_model="ollama/qwen2.5:7b", prioritizer_model="ollama/llama3.1:8b"
        ).execute(requirements)

    assert len(batch_state.classified_requirements) == _N - 1
    assert len(streaming_state.classified_requirements) == _N - 1
    assert "req-03" not in _content_by_id(batch_state)
    assert "req-03" not in _content_by_id(streaming_state)
    assert _content_by_id(batch_state) == _content_by_id(streaming_state)

    streaming_modes = {r["mode"] for r in streaming_state.metrics.get("failure_detections", [])}
    assert "schema_invalid" in streaming_modes
