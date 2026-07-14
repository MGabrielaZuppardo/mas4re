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
    """Build and compile the MAS pipeline.

    Flow::

        START -> classifier -> prioritizer -> cross_check -> END

    Agents are injected (ADR-002), keeping the graph decoupled from
    LLM construction and model selection. The shared state is the
    Pydantic ``PipelineState`` (ADR-005). ``cross_check`` (ADR-003) is a
    pure post-processing node (no LLM call) that detects inter-agent
    conflicts and stores them in ``state.metrics["inter_agent_conflicts"]``
    — a signal that only exists for this multi-agent architecture.
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
