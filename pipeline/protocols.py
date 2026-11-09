"""Graphs of the coordination protocols (the experimental factor of the dissertation).

The agents and the cross-check are the same in every protocol; what changes is who may talk to
whom, and when::

    P0 (fixed)    classifier -> prioritizer -> cross_check
    P2 (peers)    classifier -> prioritizer -> classifier_answers -> prioritizer_reconsiders
                  -> cross_check
    P3 (mediated) classifier -> prioritizer -> mediator -> prioritizer_answers -> cross_check

The topology is static: a turn in which nobody has anything pending does nothing. A single round of
exchange in P2 and P3 keeps the cost bounded and the protocols comparable.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.agentic.classifier import AgenticClassifier
from agents.agentic.prioritizer import AgenticPrioritizer
from domain.enums import CoordinationProtocol, Lang
from domain.models import PipelineState
from pipeline.graph import build_pipeline_graph
from pipeline.mediation import DEFAULT_MAX_REQUESTS, MUST_SHARE_LIMIT
from pipeline.nodes.classifier_node import classifier_node
from pipeline.nodes.cross_check_node import cross_check_node
from pipeline.nodes.prioritizer_node import prioritizer_node
from pipeline.nodes.protocol_nodes import (
    classifier_answers_node,
    mediator_node,
    prioritizer_answers_node,
    prioritizer_reconsiders_node,
)


def _chain(state_graph: StateGraph, nodes: list[str]) -> CompiledStateGraph:
    state_graph.set_entry_point(nodes[0])
    for current, following in zip(nodes, nodes[1:], strict=False):
        state_graph.add_edge(current, following)
    state_graph.add_edge(nodes[-1], END)
    return state_graph.compile()


def build_protocol_graph(
    protocol: CoordinationProtocol,
    classifier: AgenticClassifier,
    prioritizer: AgenticPrioritizer,
    lang: Lang = Lang.PT,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    must_limit: float = MUST_SHARE_LIMIT,
) -> CompiledStateGraph:
    if protocol is CoordinationProtocol.FIXED:
        return build_pipeline_graph(classifier, prioritizer)

    graph: StateGraph = StateGraph(PipelineState)
    graph.add_node("classifier", partial(classifier_node, agent=classifier))
    graph.add_node("prioritizer", partial(prioritizer_node, agent=prioritizer))
    graph.add_node("cross_check", cross_check_node)

    if protocol is CoordinationProtocol.PEERS:
        graph.add_node("classifier_answers", partial(classifier_answers_node, agent=classifier))
        graph.add_node(
            "prioritizer_reconsiders", partial(prioritizer_reconsiders_node, agent=prioritizer)
        )
        return _chain(
            graph,
            [
                "classifier",
                "prioritizer",
                "classifier_answers",
                "prioritizer_reconsiders",
                "cross_check",
            ],
        )

    graph.add_node(
        "mediator",
        partial(mediator_node, lang=lang, max_requests=max_requests, must_limit=must_limit),
    )
    graph.add_node("prioritizer_answers", partial(prioritizer_answers_node, agent=prioritizer))
    return _chain(
        graph, ["classifier", "prioritizer", "mediator", "prioritizer_answers", "cross_check"]
    )
