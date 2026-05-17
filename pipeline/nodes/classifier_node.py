from __future__ import annotations

from agents.classifier import ClassificationAgent
from domain.models import PipelineState


def classifier_node(
    state: PipelineState,
    agent: ClassificationAgent,
) -> PipelineState:
    """LangGraph node: classify raw requirements via the injected agent.

    The agent is injected (ADR-002) so the graph stays decoupled from
    LLM construction. Returns the mutated PipelineState with
    ``classified_requirements`` populated.
    """
    return agent.run(state)
