"""End-to-end smoke test for the minimal LangGraph pipeline.

The LLM is fully mocked, so this runs without Ollama and is
deterministic. It validates the full flow:
raw -> classifier_node -> prioritizer_node -> prioritized.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from domain.models import PipelineState, Requirement
from pipeline.graph import build_pipeline_graph

_CLASSIFIER_JSON = (
    '{"requirement_type": "NF", "nfr_category": "SE", "confidence": 0.9,'
    ' "justification": "Security-related requirement."}'
)
_PRIORITIZER_JSON = (
    '{"priority": "M", "priority_score": 1.0, "priority_rank": 1,'
    ' "justification": "Critical for the system."}'
)


def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


@pytest.fixture
def sample_requirements() -> list[Requirement]:
    return [
        Requirement(id=f"req-{i:02d}", text=f"The system must securely handle case {i}.")
        for i in range(10)
    ]


def test_pipeline_e2e_smoke(sample_requirements: list[Requirement]) -> None:
    with (
        patch("agents.classifier.build_llm", return_value=_mock_llm(_CLASSIFIER_JSON)),
        patch("agents.prioritizer.build_llm", return_value=_mock_llm(_PRIORITIZER_JSON)),
    ):
        classifier = ClassificationAgent(model="ollama/qwen2.5:7b")
        prioritizer = PrioritizationAgent(model="ollama/llama3.1:8b")
        graph = build_pipeline_graph(classifier, prioritizer)

        initial = PipelineState(raw_requirements=sample_requirements)
        result = graph.invoke(initial)

    # LangGraph returns the final state as a dict keyed by field names.
    prioritized = result["prioritized_requirements"]
    classified = result["classified_requirements"]

    assert len(classified) == 10
    assert len(prioritized) == 10
    # Global ranking applied by the prioritizer.
    ranks = sorted(r.priority_rank for r in prioritized)
    assert ranks == list(range(1, 11))
