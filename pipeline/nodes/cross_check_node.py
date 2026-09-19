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

    Also tracks the mediated-pipeline pass count and a per-pass
    conflict snapshot (ADR-009), even for the fixed-flow graph
    (pipeline/graph.py::build_pipeline_graph) where nothing reads them
    — this node is shared by both conditions on purpose, so Condition A
    stays behaviorally unchanged (single-factor experiment) while
    Condition B's router (build_mediated_pipeline_graph) can rely on
    the same state.
    """
    conflicts = check_requirements(state.prioritized_requirements)
    state.metrics["inter_agent_conflicts"] = [c.model_dump(mode="json") for c in conflicts]

    current_pass = state.retry_counts.get("_pass", 0) + 1
    state.retry_counts["_pass"] = current_pass
    state.metrics.setdefault("mediation_passes", []).append(
        {"pass": current_pass, "conflicts": [c.requirement_id for c in conflicts]}
    )
    return state
