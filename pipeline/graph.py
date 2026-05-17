from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from domain.models import PipelineState
from pipeline.nodes.classifier_node import classifier_node
from pipeline.nodes.prioritizer_node import prioritizer_node


def build_pipeline_graph(
    classifier: ClassificationAgent,
    prioritizer: PrioritizationAgent,
) -> CompiledStateGraph:
    """Build and compile the minimal MAS pipeline.

    Flow::

        START -> classifier -> prioritizer -> END

    Agents are injected (ADR-002), keeping the graph decoupled from
    LLM construction and model selection. The shared state is the
    Pydantic ``PipelineState`` (ADR-005).
    """
    graph: StateGraph = StateGraph(PipelineState)

    graph.add_node("classifier", partial(classifier_node, agent=classifier))
    graph.add_node("prioritizer", partial(prioritizer_node, agent=prioritizer))

    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "prioritizer")
    graph.add_edge("prioritizer", END)

    return graph.compile()