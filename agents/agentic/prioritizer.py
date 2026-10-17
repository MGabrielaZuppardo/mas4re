from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agents.agentic.base import AgenticAgent
from agents.agentic.runtime import DEFAULT_MAX_STEPS, AgentRuntime, EpisodeTask
from agents.agentic.toolkit import (
    BacklogView,
    make_backlog_overview_tool,
    make_recall_tool,
    make_reclassification_request_tool,
    make_terminal_tool,
)
from agents.base import rank_by_priority
from agents.memory import EpisodicMemory
from domain.agent_actions import Belief, Goal
from domain.enums import Lang, MoSCoWPriority
from domain.messages import (
    ClassificationReport,
    Message,
    Performative,
    ReclassificationRequest,
    inbox_of,
    latest_per_requirement,
)
from domain.models import (
    ClassifiedRequirement,
    PipelineState,
    PrioritizationOutput,
    PrioritizedRequirement,
    Requirement,
)
from evaluation.failure_detectors import DetectionContext
from llm.factory import build_llm
from prompts.v2.agent_prioritization import (
    build_prioritizer_percept,
    build_prioritizer_system_prompt,
    describe_classification,
    prioritizer_goal,
)

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter

SUBMIT_PRIORITY = "submit_priority"

logger = logging.getLogger(__name__)


class PrioritySubmission(BaseModel):
    priority: MoSCoWPriority
    priority_score: float | None = Field(default=None, ge=0.0, le=1.0)
    justification: str = ""


@dataclass(frozen=True)
class Handoff:
    """A requirement as it reaches the prioritizer: the classifier's report and nothing else."""

    id: str
    message_id: str
    report: ClassificationReport


class AgenticPrioritizer(AgenticAgent[Handoff, PrioritizedRequirement]):
    """Prioritizes as an agent. What it knows about a requirement is the `Inform` message the
    classifier sent; it never reads the classifier's output from the shared state. It may ask
    the classifier to reconsider a classification, and the classifier may refuse."""

    name = "prioritizer"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        lang: Lang = Lang.PT,
        trace_writer: TraceWriter | None = None,
        peer: str = "classifier",
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        super().__init__(model=model, temperature=temperature, trace_writer=trace_writer)
        self._lang = lang
        self._peer = peer
        self._view: BacklogView | None = None
        self._outbox: list[Message] = []
        self._active = threading.local()
        self._goal = Goal(description=prioritizer_goal(lang))
        self._system_prompt = build_prioritizer_system_prompt(lang)
        self._runtime = AgentRuntime(
            build_llm(model, temperature),
            [
                make_recall_tool(lambda: self._memory),
                make_backlog_overview_tool(lambda: self._view),
                make_reclassification_request_tool(self._request_reclassification),
                make_terminal_tool(
                    SUBMIT_PRIORITY,
                    "Submit the final priority of the requirement. Ends your work on it.",
                    PrioritySubmission,
                ),
            ],
            frozenset({SUBMIT_PRIORITY}),
            max_steps,
        )
        logger.info("AgenticPrioritizer inicializado | model=%s | lang=%s", model, lang.value)

    def _runtimes(self) -> list[AgentRuntime]:
        return [self._runtime]

    def run(self, state: PipelineState) -> PipelineState:
        prioritized = self.prioritize_batch(self._handoffs(state))
        sources = {requirement.id: requirement for requirement in state.raw_requirements}
        state.prioritized_requirements = [
            self._with_provenance(item, sources.get(item.id)) for item in prioritized
        ]
        state.messages.extend(self._outbox)
        self._outbox = []
        self._persist_failure_records(state)
        self._persist_episodes(state)
        return state

    def prioritize_batch(
        self,
        handoffs: list[Handoff],
        max_workers: int | None = None,
    ) -> list[PrioritizedRequirement]:
        self._memory = EpisodicMemory()
        self._view = BacklogView(
            by_type=Counter(h.report.requirement_type.value for h in handoffs),
            by_category=Counter(h.report.nfr_category for h in handoffs if h.report.nfr_category),
            memory=self._memory,
        )
        ordered = self._run_batch(handoffs, stage="prioritize", max_workers=max_workers)
        return rank_by_priority(ordered)

    def _handoffs(self, state: PipelineState) -> list[Handoff]:
        informs = inbox_of(state.messages, self.name, Performative.INFORM)
        return [
            Handoff(id=message.requirement_id, message_id=message.id, report=message.content)
            for message in latest_per_requirement(informs)
            if isinstance(message.content, ClassificationReport)
        ]

    @staticmethod
    def _with_provenance(
        item: PrioritizedRequirement, source: Requirement | None
    ) -> PrioritizedRequirement:
        """The item's origin (dataset source and metadata) comes from the environment's own
        record of the requirement, not from the classifier's message."""
        if source is None:
            return item
        return item.model_copy(update={"source": source.source, "metadata": source.metadata})

    def _request_reclassification(self, reason: str) -> str:
        self._outbox.append(
            Message(
                sender=self.name,
                receiver=self._peer,
                performative=Performative.REQUEST,
                requirement_id=self._active.requirement_id,
                content=ReclassificationRequest(reason=reason),
            )
        )
        return "Request sent to the classifier. Its answer will arrive later as a message."

    def _text_of(self, handoff: Handoff) -> str:
        report = handoff.report
        return report.text_en if self._lang is Lang.EN and report.text_en else report.text

    def _task_for(self, handoff: Handoff) -> EpisodeTask:
        report = handoff.report
        classification = describe_classification(
            report.requirement_type.value,
            report.nfr_category,
            report.confidence,
            report.justification,
            self._lang,
        )
        return EpisodeTask(
            agent=self.name,
            model=self.model,
            requirement_id=handoff.id,
            goal=self._goal,
            system_prompt=self._system_prompt,
            percept=build_prioritizer_percept(self._text_of(handoff), classification, self._lang),
            beliefs=[
                Belief(source=f"message:{handoff.message_id}", content=classification),
            ],
        )

    def _process_single(self, handoff: Handoff) -> PrioritizedRequirement:
        self._active.requirement_id = handoff.id
        submission = self._run_episode(self._runtime, self._task_for(handoff))
        assert isinstance(submission.arguments, PrioritySubmission)
        decision = submission.arguments
        output = PrioritizationOutput(
            requirement_id=handoff.id,
            priority=decision.priority,
            priority_score=(
                decision.priority_score
                if decision.priority_score is not None
                else decision.priority.score
            ),
            priority_rank=1,  # provisional: rank_by_priority ranks the whole batch afterwards
            justification=decision.justification,
        )
        self._remember(handoff, output)
        report = handoff.report
        classified = ClassifiedRequirement(
            id=handoff.id,
            text=report.text,
            text_en=report.text_en,
            requirement_type=report.requirement_type,
            nfr_category=report.nfr_category,
            confidence=report.confidence,
            justification=report.justification,
        )
        return PrioritizedRequirement.from_classified(classified, output)

    def _remember(self, handoff: Handoff, output: PrioritizationOutput) -> None:
        if self._view is not None:
            self._view.record(output.priority.value)
        if self._memory is not None:
            self._memory.add(
                handoff.id,
                self._text_of(handoff),
                output_summary=output.priority.value,
                justification=output.justification,
            )

    def _detection_context(
        self,
        item: Handoff,
        output: PrioritizedRequirement | None,
        stage: str,
        parsed_ok: bool,
    ) -> DetectionContext | None:
        return DetectionContext(
            requirement_id=item.id,
            stage=stage,
            requirement_text=self._text_of(item),
            parsed_ok=parsed_ok,
            justification=output.priority_justification if output is not None else "",
        )
