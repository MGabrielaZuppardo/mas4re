from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from agents.base import rank_by_priority
from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from domain.models import PipelineState
from pipeline.nodes.cross_check_node import cross_check_node
from pipeline.nodes.process_item_node import process_item_node


def _dispatch_items(state: PipelineState) -> list[Send]:
    """Fan-out edge: one independent process_item branch per requirement.

    This is what makes the pipeline stream per item instead of running two
    blocking batches (classify-all, then prioritize-all) — each Send target
    starts as soon as it's dispatched and finishes classify -> prioritize on
    its own item, unblocked by every other item's progress.
    """
    return [Send("process_item", {"requirement": r}) for r in state.raw_requirements]


def _aggregate_node(state: PipelineState) -> dict:
    """Fan-in: global ranking and failure-record persistence only make sense
    once every process_item branch has finished.

    Branches complete in whatever order the LLM calls happen to finish, not
    input order, so classified/prioritized_requirements are re-sorted back to
    the original requirement order first -- otherwise rank_by_priority's
    stable sort could break priority_score ties differently than the batch
    pipeline (pipeline/graph.py) would for the same input.

    Returns a partial update dict (NOT the full PipelineState object) that
    writes the finalized lists into the plain, non-reducer public fields
    (classified_requirements/prioritized_requirements) as an overwrite —
    reading from the reducer-accumulated scratch fields (stream_classified/
    stream_prioritized) but never re-returning them, so they aren't doubled
    by cross_check_node's subsequent full-state return (harmless either way,
    since nothing reads the scratch fields again after this point).
    """
    order = {r.id: i for i, r in enumerate(state.raw_requirements)}
    classified = sorted(state.stream_classified, key=lambda c: order.get(c.id, 0))
    prioritized = sorted(state.stream_prioritized, key=lambda p: order.get(p.id, 0))
    prioritized = rank_by_priority(prioritized)

    metrics = dict(state.metrics)
    if state.failure_records:
        metrics.setdefault("failure_detections", []).extend(
            r.model_dump(mode="json") for r in state.failure_records
        )

    return {
        "classified_requirements": classified,
        "prioritized_requirements": prioritized,
        "metrics": metrics,
    }


def build_streaming_pipeline_graph(
    classifier: ClassificationAgent,
    prioritizer: PrioritizationAgent,
) -> CompiledStateGraph:
    """Build and compile the per-item streaming variant of the MAS pipeline.

    Flow::

        START --(Send per requirement)--> process_item (N parallel branches)
        process_item -> aggregate -> cross_check -> END

    Preserves the same invariants as the batch pipeline (pipeline/graph.py):
    a fully validated ClassifiedRequirement crosses the classifier/prioritizer
    boundary, per-item confidence routing is untouched, and a failing item is
    dropped without affecting any other item's result — only the scheduling
    changes (per item instead of per batch). See experiments/strategy.py::
    PipelineStreamingStrategy for how this is wired into an experiment run,
    and pipeline/graph.py for the (unmodified) batch counterpart.
    """
    graph: StateGraph = StateGraph(PipelineState)

    graph.add_node(
        "process_item", partial(process_item_node, classifier=classifier, prioritizer=prioritizer)
    )
    graph.add_node("aggregate", _aggregate_node)
    graph.add_node("cross_check", cross_check_node)

    graph.add_conditional_edges(START, _dispatch_items, ["process_item"])
    graph.add_edge("process_item", "aggregate")
    graph.add_edge("aggregate", "cross_check")
    graph.add_edge("cross_check", END)

    return graph.compile()
