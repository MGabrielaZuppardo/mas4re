from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from agents.baseline import BaselineAgent
from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from agents.two_call_baseline import TwoCallBaselineAgent
from config.settings import settings
from domain.enums import Lang
from domain.models import PipelineState, Requirement
from pipeline.graph import build_pipeline_graph
from pipeline.graph_streaming import build_streaming_pipeline_graph

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter


def _coerce_state(result: object) -> PipelineState:
    """LangGraph may return a dict; normalize back to PipelineState."""
    if isinstance(result, PipelineState):
        return result
    if isinstance(result, dict):
        return PipelineState.model_validate(result)
    raise TypeError(f"Unexpected pipeline result type: {type(result)!r}")


class OrchestrationStrategy(ABC):
    """Strategy that turns raw requirements into a processed PipelineState.

    Isolates the architectural variable for SQ2: every strategy receives
    the same requirements and returns the same state shape, so the only
    thing that differs between runs is the orchestration design.

    The optional trace_writer parameter in execute() is injected by
    ExperimentRunner so all agent calls are recorded in the run's JSONL.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def execute(
        self,
        requirements: list[Requirement],
        trace_writer: TraceWriter | None = None,
    ) -> PipelineState: ...


class BaselineStrategy(OrchestrationStrategy):
    """Single-agent baseline: classify + prioritize in one LLM call."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        nfr_categories: list[tuple[str, str]] | None = None,
        lang: Lang = Lang.PT,
    ) -> None:
        self._agent = BaselineAgent(
            model=model,
            temperature=temperature,
            nfr_categories=nfr_categories,
            lang=lang,
        )

    @property
    def name(self) -> str:
        return "baseline"

    def execute(
        self,
        requirements: list[Requirement],
        trace_writer: TraceWriter | None = None,
    ) -> PipelineState:
        self._agent._trace = trace_writer
        state = PipelineState(raw_requirements=requirements)
        return self._agent.run(state)


class PipelineStrategy(OrchestrationStrategy):
    """Multi-agent pipeline: classifier -> prioritizer via LangGraph."""

    def __init__(
        self,
        classifier_model: str,
        prioritizer_model: str,
        temperature: float = 0.0,
        nfr_categories: list[tuple[str, str]] | None = None,
        lang: Lang = Lang.PT,
    ) -> None:
        self._classifier = ClassificationAgent(
            model=classifier_model,
            temperature=temperature,
            nfr_categories=nfr_categories,
            lang=lang,
        )
        self._prioritizer = PrioritizationAgent(
            model=prioritizer_model,
            temperature=temperature,
            lang=lang,
        )
        self._graph = build_pipeline_graph(self._classifier, self._prioritizer)

    @property
    def name(self) -> str:
        return "pipeline"

    def execute(
        self,
        requirements: list[Requirement],
        trace_writer: TraceWriter | None = None,
    ) -> PipelineState:
        self._classifier._trace = trace_writer
        self._prioritizer._trace = trace_writer
        state = PipelineState(raw_requirements=requirements)
        return _coerce_state(self._graph.invoke(state))


class PipelineStreamingStrategy(OrchestrationStrategy):
    """Multi-agent pipeline, per-item streaming variant of PipelineStrategy.

    Same classifier -> prioritizer typed handoff, per-item confidence
    routing, and failure isolation as PipelineStrategy — only the
    scheduling differs: each requirement is dispatched independently (via
    langgraph.types.Send, see pipeline/graph_streaming.py) so it can be
    prioritized as soon as its own classification finishes, instead of
    waiting for the whole batch to classify first. Opt-in and separate from
    PipelineStrategy on purpose: it isn't wired into scripts/run_grid.py or
    any existing grid condition, so no existing run is affected by this
    strategy's introduction.
    """

    def __init__(
        self,
        classifier_model: str,
        prioritizer_model: str,
        temperature: float = 0.0,
        nfr_categories: list[tuple[str, str]] | None = None,
        lang: Lang = Lang.PT,
    ) -> None:
        self._classifier = ClassificationAgent(
            model=classifier_model,
            temperature=temperature,
            nfr_categories=nfr_categories,
            lang=lang,
        )
        self._prioritizer = PrioritizationAgent(
            model=prioritizer_model,
            temperature=temperature,
            lang=lang,
        )
        self._graph = build_streaming_pipeline_graph(self._classifier, self._prioritizer)

    @property
    def name(self) -> str:
        return "pipeline_streaming"

    def execute(
        self,
        requirements: list[Requirement],
        trace_writer: TraceWriter | None = None,
    ) -> PipelineState:
        self._classifier._trace = trace_writer
        self._prioritizer._trace = trace_writer
        state = PipelineState(raw_requirements=requirements)
        result = self._graph.invoke(state, config={"max_concurrency": settings.max_workers})
        return _coerce_state(result)


class TwoCallBaselineStrategy(OrchestrationStrategy):
    """Ablation: two sequential LLM calls with specialised prompts,
    no typed inter-agent state, no confidence-based routing.

    Uses the same prompt templates as the pipeline (ClassificationAgent
    + PrioritizationAgent) but passes intermediate results via a plain
    Python dataclass instead of a validated PipelineState.  This isolates
    the contribution of the typed state contract from the prompt design.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        nfr_categories: list[tuple[str, str]] | None = None,
        lang: Lang = Lang.PT,
    ) -> None:
        self._agent = TwoCallBaselineAgent(
            model=model,
            temperature=temperature,
            nfr_categories=nfr_categories,
            lang=lang,
        )

    @property
    def name(self) -> str:
        return "two_call_baseline"

    def execute(
        self,
        requirements: list[Requirement],
        trace_writer: TraceWriter | None = None,
    ) -> PipelineState:
        self._agent._trace = trace_writer
        state = PipelineState(raw_requirements=requirements)
        return self._agent.run(state)
