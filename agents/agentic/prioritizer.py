from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from agents.agentic.base import AgentEpisodeError, AgenticAgent
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
from domain.agent_actions import Belief, EpisodePhase, Goal
from domain.enums import CoordinationProtocol, Lang, MoSCoWPriority
from domain.messages import (
    ClassificationReport,
    Message,
    Performative,
    PriorityReport,
    ReclassificationRequest,
    ReprioritizationRequest,
    RequestReply,
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
    build_prioritizer_reconsideration_system_prompt,
    build_prioritizer_reply_system_prompt,
    build_prioritizer_system_prompt,
    build_priority_reply_percept,
    build_reconsideration_percept,
    describe_classification,
    describe_peer_answer,
    describe_priority,
    prioritizer_goal,
    priority_reply_goal,
    reconsideration_goal,
)

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter

SUBMIT_PRIORITY = "submit_priority"
REPLY_TO_REQUEST = "reply_to_request"

logger = logging.getLogger(__name__)


class PrioritySubmission(BaseModel):
    priority: MoSCoWPriority
    priority_score: float | None = Field(default=None, ge=0.0, le=1.0)
    justification: str = ""


class PriorityReplySubmission(BaseModel):
    decision: Literal["agree", "refuse"]
    reason: str
    priority: MoSCoWPriority | None = None
    priority_score: float | None = Field(default=None, ge=0.0, le=1.0)
    justification: str = ""

    @model_validator(mode="after")
    def _agreeing_needs_the_revised_priority(self) -> PriorityReplySubmission:
        if self.decision == "agree" and self.priority is None:
            raise ValueError("agreeing requires the revised priority")
        return self


@dataclass(frozen=True)
class Handoff:
    """A requirement as it reaches the prioritizer: the classifier's report and nothing else."""

    id: str
    message_id: str
    report: ClassificationReport


class AgenticPrioritizer(AgenticAgent[Handoff, PrioritizedRequirement]):
    """Prioritizes as an agent. What it knows about a requirement is the `Inform` message the
    classifier sent; it never reads the classifier's output from the shared state.

    The coordination protocol decides what else it can do. Under PEERS it may ask the classifier
    to reconsider a classification (the classifier may refuse) and then decides the priority again
    with the answer. Under MEDIATED it may be asked by the mediator to reconsider a priority, and
    agrees or refuses. Under FIXED it has neither: its action space holds no channel back.
    """

    name = "prioritizer"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        lang: Lang = Lang.PT,
        trace_writer: TraceWriter | None = None,
        peer: str = "classifier",
        max_steps: int = DEFAULT_MAX_STEPS,
        protocol: CoordinationProtocol = CoordinationProtocol.FIXED,
    ) -> None:
        super().__init__(model=model, temperature=temperature, trace_writer=trace_writer)
        self._lang = lang
        self._peer = peer
        self._protocol = protocol
        self._view: BacklogView | None = None
        self._outbox: list[Message] = []
        self._active = threading.local()
        self._goal = Goal(description=prioritizer_goal(lang))
        peers = protocol is CoordinationProtocol.PEERS
        self._system_prompt = build_prioritizer_system_prompt(lang, peer_requests=peers)

        llm = build_llm(model, temperature)
        evidence_tools = [
            make_recall_tool(lambda: self._memory),
            make_backlog_overview_tool(lambda: self._view),
        ]
        submit = make_terminal_tool(
            SUBMIT_PRIORITY,
            "Submit the final priority of the requirement. Ends your work on it.",
            PrioritySubmission,
        )
        request_tools = (
            [make_reclassification_request_tool(self._request_reclassification)] if peers else []
        )
        self._runtime = AgentRuntime(
            llm, [*evidence_tools, *request_tools, submit], frozenset({SUBMIT_PRIORITY}), max_steps
        )
        self._reconsider_runtime = (
            AgentRuntime(llm, [*evidence_tools, submit], frozenset({SUBMIT_PRIORITY}), max_steps)
            if peers
            else None
        )
        self._reply_runtime = (
            AgentRuntime(
                llm,
                [
                    *evidence_tools,
                    make_terminal_tool(
                        REPLY_TO_REQUEST,
                        "Answer the request: agree, with the revised priority, or refuse, with "
                        "the reason. Ends your work on it.",
                        PriorityReplySubmission,
                    ),
                ],
                frozenset({REPLY_TO_REQUEST}),
                max_steps,
            )
            if protocol is CoordinationProtocol.MEDIATED
            else None
        )
        logger.info(
            "AgenticPrioritizer inicializado | model=%s | lang=%s | protocol=%s",
            model,
            lang.value,
            protocol.value,
        )

    def _runtimes(self) -> list[AgentRuntime]:
        runtimes = [self._runtime, self._reconsider_runtime, self._reply_runtime]
        return [runtime for runtime in runtimes if runtime is not None]

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

    # ── PEERS: deciding again after the classifier answered ────────────────────────────

    def reconsider(self, state: PipelineState) -> PipelineState:
        """Decide once more the priority of every item whose reclassification request the
        classifier answered (agreeing or refusing). One round: the reconsideration runtime has
        no way to ask again."""
        if self._reconsider_runtime is None:
            raise ValueError("reconsider() needs the PEERS protocol")
        for answer in self._unreconsidered_answers(state):
            item = next(
                (p for p in state.prioritized_requirements if p.id == answer.requirement_id), None
            )
            if item is not None:
                self._reconsider(state, item, answer)
        state.prioritized_requirements = rank_by_priority(state.prioritized_requirements)
        self._persist_episodes(state)
        return state

    def _unreconsidered_answers(self, state: PipelineState) -> list[Message]:
        mine = {
            m.id
            for m in state.messages
            if m.sender == self.name and m.performative is Performative.REQUEST
        }
        settled = {
            m.in_reply_to
            for m in state.messages
            if m.sender == self.name and m.performative is Performative.INFORM and m.in_reply_to
        }
        return [
            m
            for m in inbox_of(state.messages, self.name)
            if m.in_reply_to in mine
            and m.id not in settled
            and m.performative in (Performative.AGREE, Performative.REFUSE)
        ]

    def _reconsider(
        self, state: PipelineState, item: PrioritizedRequirement, answer: Message
    ) -> None:
        assert self._reconsider_runtime is not None
        content = answer.content
        revised = content.revised if isinstance(content, RequestReply) else None
        current = self._with_report(item, revised) if revised is not None else item
        try:
            submission = self._run_episode(
                self._reconsider_runtime, self._reconsideration_task(current, answer)
            )
        except AgentEpisodeError:
            return
        assert isinstance(submission.arguments, PrioritySubmission)
        decided = self._with_decision(current, submission.arguments)
        state.prioritized_requirements = [
            decided if p.id == decided.id else p for p in state.prioritized_requirements
        ]
        state.messages.append(
            Message(
                sender=self.name,
                receiver=self._peer,
                performative=Performative.INFORM,
                requirement_id=decided.id,
                content=self._priority_report(decided),
                in_reply_to=answer.id,
            )
        )

    def _reconsideration_task(self, item: PrioritizedRequirement, answer: Message) -> EpisodeTask:
        assert isinstance(answer.content, RequestReply)
        previous = self._describe_decision(item)
        decision = "agree" if answer.performative is Performative.AGREE else "refuse"
        peer_answer = describe_peer_answer(decision, answer.content.reason, self._lang)
        return EpisodeTask(
            agent=self.name,
            model=self.model,
            requirement_id=item.id,
            goal=Goal(description=reconsideration_goal(self._lang)),
            system_prompt=build_prioritizer_reconsideration_system_prompt(self._lang),
            percept=build_reconsideration_percept(
                self._item_text(item),
                self._describe_classification(item),
                previous,
                peer_answer,
                self._lang,
            ),
            beliefs=[
                Belief(source="self", content=previous),
                Belief(source=f"message:{answer.id}", content=peer_answer),
            ],
            phase=EpisodePhase.RECONSIDERATION,
        )

    # ── MEDIATED: answering the mediator ───────────────────────────────────────────────

    def answer_requests(self, state: PipelineState) -> PipelineState:
        """Answer the reprioritization requests addressed to this agent that it has not yet
        answered. For each one it decides to agree, revising the priority, or to refuse."""
        if self._reply_runtime is None:
            raise ValueError("answer_requests() needs the MEDIATED protocol")
        answered = {m.in_reply_to for m in state.messages if m.sender == self.name}
        for request in inbox_of(state.messages, self.name, Performative.REQUEST):
            if request.id not in answered:
                self._answer(state, request)
        state.prioritized_requirements = rank_by_priority(state.prioritized_requirements)
        self._persist_episodes(state)
        return state

    def _answer(self, state: PipelineState, request: Message) -> None:
        assert self._reply_runtime is not None
        item = next(
            (p for p in state.prioritized_requirements if p.id == request.requirement_id), None
        )
        if item is None or not isinstance(request.content, ReprioritizationRequest):
            state.messages.append(self._reply(request, Performative.FAILURE, "Cannot be answered."))
            return
        try:
            submission = self._run_episode(self._reply_runtime, self._reply_task(item, request))
        except AgentEpisodeError:
            state.messages.append(
                self._reply(request, Performative.FAILURE, "No answer within the step budget.")
            )
            return
        assert isinstance(submission.arguments, PriorityReplySubmission)
        answer = submission.arguments
        if answer.decision == "refuse" or answer.priority is None:
            state.messages.append(self._reply(request, Performative.REFUSE, answer.reason))
            return
        decided = self._with_decision(
            item,
            PrioritySubmission(
                priority=answer.priority,
                priority_score=answer.priority_score,
                justification=answer.justification,
            ),
        )
        state.prioritized_requirements = [
            decided if p.id == decided.id else p for p in state.prioritized_requirements
        ]
        state.messages.append(
            self._reply(request, Performative.AGREE, answer.reason, self._priority_report(decided))
        )

    def _reply(
        self,
        request: Message,
        performative: Performative,
        reason: str,
        revised: PriorityReport | None = None,
    ) -> Message:
        return Message(
            sender=self.name,
            receiver=request.sender,
            performative=performative,
            requirement_id=request.requirement_id,
            content=RequestReply(reason=reason, revised_priority=revised),
            in_reply_to=request.id,
        )

    def _reply_task(self, item: PrioritizedRequirement, request: Message) -> EpisodeTask:
        assert isinstance(request.content, ReprioritizationRequest)
        previous = self._describe_decision(item)
        asked = f"{request.content.reason} {request.content.evidence}".strip()
        return EpisodeTask(
            agent=self.name,
            model=self.model,
            requirement_id=item.id,
            goal=Goal(description=priority_reply_goal(self._lang)),
            system_prompt=build_prioritizer_reply_system_prompt(self._lang),
            percept=build_priority_reply_percept(
                self._item_text(item),
                self._describe_classification(item),
                previous,
                request.content.reason,
                request.content.evidence,
                self._lang,
            ),
            beliefs=[
                Belief(source="self", content=previous),
                Belief(source=f"message:{request.id}", content=asked),
            ],
            phase=EpisodePhase.REPLY,
        )

    # ── shared helpers ─────────────────────────────────────────────────────────────────

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

    @staticmethod
    def _with_report(
        item: PrioritizedRequirement, report: ClassificationReport
    ) -> PrioritizedRequirement:
        return item.model_copy(
            update={
                "requirement_type": report.requirement_type,
                "nfr_category": report.nfr_category,
                "confidence": report.confidence,
                "justification": report.justification,
            }
        )

    @staticmethod
    def _with_decision(
        item: PrioritizedRequirement, decision: PrioritySubmission
    ) -> PrioritizedRequirement:
        score = (
            decision.priority_score
            if decision.priority_score is not None
            else decision.priority.score
        )
        return item.model_copy(
            update={
                "priority": decision.priority,
                "priority_score": score,
                "priority_justification": decision.justification,
            }
        )

    @staticmethod
    def _priority_report(item: PrioritizedRequirement) -> PriorityReport:
        assert item.priority is not None and item.priority_score is not None
        return PriorityReport(
            priority=item.priority,
            priority_score=item.priority_score,
            justification=item.priority_justification,
        )

    def _describe_decision(self, item: PrioritizedRequirement) -> str:
        assert item.priority is not None and item.priority_score is not None
        return describe_priority(
            item.priority.value, item.priority_score, item.priority_justification, self._lang
        )

    def _describe_classification(self, item: PrioritizedRequirement) -> str:
        return describe_classification(
            item.requirement_type.value,
            item.nfr_category,
            item.confidence,
            item.justification,
            self._lang,
        )

    def _item_text(self, item: PrioritizedRequirement) -> str:
        return item.text_en if self._lang is Lang.EN and item.text_en else item.text

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
