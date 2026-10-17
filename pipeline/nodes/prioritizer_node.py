from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from domain.models import PipelineState


def prioritizer_node(
    state: PipelineState,
    agent: BaseAgent[Any, Any],
) -> PipelineState:
    """LangGraph node: prioritize classified requirements via the agent.

    The agent is injected (ADR-002). Returns the mutated PipelineState
    with ``prioritized_requirements`` populated and globally ranked.
    """
    return agent.run(state)
