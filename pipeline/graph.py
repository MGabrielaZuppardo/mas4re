from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from domain.models import PipelineState
from pipeline.nodes.classifier_node import classifier_node
from pipeline.nodes.cross_check_node import cross_check_node
from pipeline.nodes.prioritizer_node import prioritizer_node


def build_pipeline_graph(
    classifier: ClassificationAgent,
    prioritizer: PrioritizationAgent,
) -> CompiledStateGraph:
    """Build and compile the minimal MAS pipeline.

    Flow::

        START -> classifier -> prioritizer -> cross_check -> END

    Agents are injected (ADR-002), keeping the graph decoupled from
    LLM construction and model selection. The shared state is the
    Pydantic ``PipelineState`` (ADR-005). cross_check is a pure
    post-processing node (no LLM) that flags inter-agent conflicts for
    SQ3 analysis (ADR-003).
    """
    graph: StateGraph = StateGraph(PipelineState)

    graph.add_node("classifier", partial(classifier_node, agent=classifier))
    graph.add_node("prioritizer", partial(prioritizer_node, agent=prioritizer))
    graph.add_node("cross_check", cross_check_node)

    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "prioritizer")
    graph.add_edge("prioritizer", "cross_check")
    graph.add_edge("cross_check", END)

    return graph.compile()


_MAX_PASSES = 2


def _route_after_cross_check(state: PipelineState) -> str:
    """Condition B's router: resend the whole batch once if cross_check
    flagged a conflict, otherwise finalize.

    Reads state only (no mutation) — LangGraph only persists state changes
    returned by real nodes, never by a conditional-edge function, so the
    pass counter is incremented in cross_check_node (a real node), not here.
    """
    has_conflict = bool(state.metrics.get("inter_agent_conflicts"))
    if has_conflict and state.retry_counts.get("_pass", 0) < _MAX_PASSES:
        return "classifier"
    return "finalize_mediation"


def finalize_mediation_node(state: PipelineState) -> PipelineState:
    """Terminal node for Condition B: summarize the mediation outcome.

    Compares the conflict set from the first pass against the last pass to
    compute how many conflicts were resolved by the retry, and stores the
    summary in state.metrics["mediation"] for SQ3/RQ analysis.
    """
    passes = state.metrics.get("mediation_passes", [])
    first_conflicts = set(passes[0]["conflicts"]) if passes else set()
    last_conflicts = set(passes[-1]["conflicts"]) if passes else set()
    resolved = first_conflicts - last_conflicts if len(passes) > 1 else set()

    state.metrics["mediation"] = {
        "conflicts_detected": len(first_conflicts),
        "resent": int(len(passes) > 1),
        "resolved_after_retry": len(resolved),
        "passes": len(passes),
    }
    return state


def build_mediated_pipeline_graph(
    classifier: ClassificationAgent,
    prioritizer: PrioritizationAgent,
) -> CompiledStateGraph:
    """Build and compile Condition B: mediated coordination via conditional
    routing.

    Flow::

        START -> classifier -> prioritizer -> cross_check
                                                  |
                                     conflict? and pass < 2
                                          /              \\
                                      yes: classifier   no: finalize_mediation -> END

    Same nodes and agents as Condition A (build_pipeline_graph) — the only
    difference between the two experimental conditions is this conditional
    edge, keeping the coordination strategy the single manipulated variable
    (ADR-009). cross_check_node is shared unmodified between both graphs.
    """
    graph: StateGraph = StateGraph(PipelineState)

    graph.add_node("classifier", partial(classifier_node, agent=classifier))
    graph.add_node("prioritizer", partial(prioritizer_node, agent=prioritizer))
    graph.add_node("cross_check", cross_check_node)
    graph.add_node("finalize_mediation", finalize_mediation_node)

    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "prioritizer")
    graph.add_edge("prioritizer", "cross_check")
    graph.add_conditional_edges(
        "cross_check", _route_after_cross_check, ["classifier", "finalize_mediation"]
    )
    graph.add_edge("finalize_mediation", END)

    return graph.compile()
