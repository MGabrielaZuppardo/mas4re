from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from agents.baseline import BaselineAgent
from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from domain.enums import Lang
from domain.models import PipelineState, Requirement
from pipeline.graph import build_pipeline_graph

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
