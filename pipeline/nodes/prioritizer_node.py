from __future__ import annotations

from agents.prioritizer import PrioritizationAgent
from domain.models import PipelineState


def prioritizer_node(
    state: PipelineState,
    agent: PrioritizationAgent,
) -> PipelineState:
    """LangGraph node: prioritize classified requirements via the agent.

    The agent is injected (ADR-002). Returns the mutated PipelineState
    with ``prioritized_requirements`` populated and globally ranked.
    """
    return agent.run(state)
