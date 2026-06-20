from __future__ import annotations

from domain.models import PipelineState
from evaluation.cross_agent_check import check_requirements


def cross_check_node(state: PipelineState) -> PipelineState:
    """LangGraph node: detect inter-agent conflicts post-prioritization.

    Pure post-processing node (no LLM): runs the deterministic
    inter-agent conflict check over the prioritized requirements and
    stores the serialized FailureRecords in
    ``state.metrics["inter_agent_conflicts"]``. Uses the existing
    metrics field to avoid a domain-model change; the runner persists
    it for SQ3 analysis (ADR-003).
    """
    conflicts = check_requirements(state.prioritized_requirements)
    state.metrics["inter_agent_conflicts"] = [c.model_dump(mode="json") for c in conflicts]
    return state
