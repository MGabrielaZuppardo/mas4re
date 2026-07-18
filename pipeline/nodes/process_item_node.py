from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluation.failure_detectors import default_chain

if TYPE_CHECKING:
    from agents.classifier import ClassificationAgent
    from agents.prioritizer import PrioritizationAgent

_DETECTOR_CHAIN = default_chain()


def process_item_node(
    item_state: dict[str, Any],
    classifier: ClassificationAgent,
    prioritizer: PrioritizationAgent,
) -> dict[str, Any]:
    """LangGraph node: classify then prioritize ONE requirement, independently
    of every other requirement in the run.

    Dispatched via langgraph.types.Send (one invocation per requirement, see
    pipeline/graph_streaming.py's fan-out edge) so an item can be prioritized
    as soon as its own classification finishes, instead of waiting for the
    whole batch — the streaming counterpart to pipeline/nodes/classifier_node.py
    + prioritizer_node.py combined.

    Reuses ClassificationAgent/PrioritizationAgent's existing
    _call_and_trace (-> _process_single, @llm_retry included) and
    _detection_context (SQ3, ADR-003) unchanged — no parsing/validation/retry
    logic is duplicated here, only the per-item orchestration differs from
    BaseAgent._run_batch.

    Returns a partial state update targeting PipelineState's scratch fields
    (stream_classified / stream_prioritized / failure_records), which are
    Annotated with operator.add so LangGraph merges every branch's
    single-item list into the shared accumulating state. _aggregate_node
    reads these once, fan-in, to compute the final classified_requirements /
    prioritized_requirements (see pipeline/graph_streaming.py for why the
    public fields themselves aren't reducer-annotated).
    """
    requirement = item_state["requirement"]
    records = []

    try:
        classified = classifier._call_and_trace("classify", requirement)
    except Exception:
        ctx = classifier._detection_context(requirement, None, "classify", parsed_ok=False)
        if ctx is not None:
            records.extend(_DETECTOR_CHAIN.run(ctx))
        return {"failure_records": records}

    ctx = classifier._detection_context(requirement, classified, "classify", parsed_ok=True)
    if ctx is not None:
        records.extend(_DETECTOR_CHAIN.run(ctx))

    try:
        prioritized = prioritizer._call_and_trace("prioritize", classified)
    except Exception:
        ctx2 = prioritizer._detection_context(classified, None, "prioritize", parsed_ok=False)
        if ctx2 is not None:
            records.extend(_DETECTOR_CHAIN.run(ctx2))
        return {"stream_classified": [classified], "failure_records": records}

    ctx2 = prioritizer._detection_context(classified, prioritized, "prioritize", parsed_ok=True)
    if ctx2 is not None:
        records.extend(_DETECTOR_CHAIN.run(ctx2))

    return {
        "stream_classified": [classified],
        "stream_prioritized": [prioritized],
        "failure_records": records,
    }
