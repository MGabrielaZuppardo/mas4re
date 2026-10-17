from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from agents.agentic.base import AgentEpisodeError, AgenticAgent
from agents.agentic.runtime import DEFAULT_MAX_STEPS, AgentRuntime, EpisodeTask
from agents.agentic.toolkit import (
    make_consistency_tool,
    make_recall_tool,
    make_terminal_tool,
    structural_issues,
)
from agents.memory import EpisodicMemory
from agents.tools import make_nfr_taxonomy_tool
from domain.agent_actions import Belief, EpisodePhase, Goal
from domain.enums import Lang, RequirementType
from domain.messages import (
    ClassificationReport,
    Message,
    Performative,
    ReclassificationRequest,
    RequestReply,
    inbox_of,
)
from domain.models import (
    ClassificationOutput,
    ClassifiedRequirement,
    PipelineState,
    Requirement,
)
from domain.nfr_normalization import NFR_ONLY_CODES, normalize_nfr_category
from evaluation.failure_detectors import DetectionContext
from llm.factory import build_llm
from prompts.v1._shared import default_nfr_categories
from prompts.v2.agent_classification import (
    build_classifier_percept,
    build_classifier_system_prompt,
    classifier_goal,
)
from prompts.v2.agent_classification_reply import (
    build_reply_percept,
    build_reply_system_prompt,
    reply_goal,
)

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter

SUBMIT_CLASSIFICATION = "submit_classification"
REPLY_TO_REQUEST = "reply_to_request"

CONFIDENCE_THRESHOLD = 0.70

logger = logging.getLogger(__name__)


class ClassificationSubmission(BaseModel):
    requirement_type: RequirementType
    nfr_category: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str = ""


class RequestReplySubmission(BaseModel):
    decision: Literal["agree", "refuse"]
    reason: str
    requirement_type: RequirementType | None = None
    nfr_category: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    justification: str = ""

    @model_validator(mode="after")
    def _agreeing_needs_the_revision(self) -> RequestReplySubmission:
        if self.decision == "agree" and (self.requirement_type is None or self.confidence is None):
            raise ValueError("agreeing requires the revised requirement_type and confidence")
        return self


class AgenticClassifier(AgenticAgent[Requirement, ClassifiedRequirement]):
    """Classifies requirements as an agent: for each one it runs an episode in which the model
    chooses its own tools and decides when its answer is ready.

    The answer goes to the peer as an `Inform` message; it is the only thing the peer learns
    about this agent's work.
    """

    name = "classifier"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        nfr_categories: list[tuple[str, str]] | None = None,
        lang: Lang = Lang.PT,
        trace_writer: TraceWriter | None = None,
        peer: str = "prioritizer",
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        super().__init__(model=model, temperature=temperature, trace_writer=trace_writer)
        self._lang = lang
        self._peer = peer
        taxonomy = nfr_categories or default_nfr_categories(lang)
        self._valid_codes = frozenset(code.upper() for code, _ in taxonomy)
        self._nfr_categories = nfr_categories
        self._goal = Goal(description=classifier_goal(lang))
        self._system_prompt = build_classifier_system_prompt(lang, nfr_categories)
        llm = build_llm(model, temperature)
        evidence_tools = [
            make_nfr_taxonomy_tool(nfr_categories, lang),
            make_recall_tool(lambda: self._memory),
            make_consistency_tool(self._valid_codes),
        ]
        self._runtime = AgentRuntime(
            llm,
            [
                *evidence_tools,
                make_terminal_tool(
                    SUBMIT_CLASSIFICATION,
                    "Submit the final classification of the requirement. Ends your work on it.",
                    ClassificationSubmission,
                ),
            ],
            frozenset({SUBMIT_CLASSIFICATION}),
            max_steps,
        )
        self._reply_runtime = AgentRuntime(
            llm,
            [
                *evidence_tools,
                make_terminal_tool(
                    REPLY_TO_REQUEST,
                    "Answer the colleague's request: agree, with the revised classification, "
                    "or refuse, with the reason. Ends your work on it.",
                    RequestReplySubmission,
                ),
            ],
            frozenset({REPLY_TO_REQUEST}),
            max_steps,
        )
        logger.info("AgenticClassifier inicializado | model=%s | lang=%s", model, lang.value)

    def _runtimes(self) -> list[AgentRuntime]:
        return [self._runtime, self._reply_runtime]

    def run(self, state: PipelineState) -> PipelineState:
        classified = self.classify_batch(state.raw_requirements)
        state.classified_requirements = classified
        state.model_used = self.model
        state.messages.extend(self._report_to_peer(item) for item in classified)
        self._persist_failure_records(state)
        self._persist_episodes(state)
        return state

    def classify_batch(
        self,
        requirements: list[Requirement],
        max_workers: int | None = None,
    ) -> list[ClassifiedRequirement]:
        self._memory = EpisodicMemory()
        return self._run_batch(requirements, stage="classify", max_workers=max_workers)

    def answer_requests(self, state: PipelineState) -> PipelineState:
        """Handle the reclassification requests addressed to this agent that it has not yet
        answered. For each one it decides to agree, revising its classification, or to refuse."""
        answered = {m.in_reply_to for m in state.messages if m.sender == self.name}
        pending = inbox_of(state.messages, self.name, Performative.REQUEST)
        for request in pending:
            if request.id not in answered:
                self._answer(state, request)
        self._persist_episodes(state)
        return state

    def _answer(self, state: PipelineState, request: Message) -> None:
        current = next(
            (c for c in state.classified_requirements if c.id == request.requirement_id), None
        )
        content = request.content
        if current is None or not isinstance(content, ReclassificationRequest):
            state.messages.append(
                self._reply(request, Performative.FAILURE, "The request cannot be answered.")
            )
            return
        try:
            submission = self._run_episode(self._reply_runtime, self._reply_task(current, request))
        except AgentEpisodeError:
            state.messages.append(
                self._reply(request, Performative.FAILURE, "No answer within the step budget.")
            )
            return
        assert isinstance(submission.arguments, RequestReplySubmission)
        answer = submission.arguments
        if answer.decision == "refuse":
            state.messages.append(self._reply(request, Performative.REFUSE, answer.reason))
            return
        revised = self._revised(current, answer)
        state.classified_requirements = [
            revised if c.id == current.id else c for c in state.classified_requirements
        ]
        state.messages.append(
            self._reply(request, Performative.AGREE, answer.reason, self._report_of(revised))
        )

    def _reply(
        self,
        request: Message,
        performative: Performative,
        reason: str,
        revised: ClassificationReport | None = None,
    ) -> Message:
        return Message(
            sender=self.name,
            receiver=request.sender,
            performative=performative,
            requirement_id=request.requirement_id,
            content=RequestReply(reason=reason, revised=revised),
            in_reply_to=request.id,
        )

    def _revised(
        self, current: ClassifiedRequirement, answer: RequestReplySubmission
    ) -> ClassifiedRequirement:
        assert answer.requirement_type is not None and answer.confidence is not None
        return current.model_copy(
            update={
                "requirement_type": answer.requirement_type,
                "nfr_category": normalize_nfr_category(answer.nfr_category),
                "confidence": answer.confidence,
                "justification": answer.justification,
            }
        )

    def _reply_task(self, current: ClassifiedRequirement, request: Message) -> EpisodeTask:
        assert isinstance(request.content, ReclassificationRequest)
        reason = request.content.reason
        return EpisodeTask(
            agent=self.name,
            model=self.model,
            requirement_id=current.id,
            goal=Goal(description=reply_goal(self._lang)),
            system_prompt=build_reply_system_prompt(self._lang, self._nfr_categories),
            percept=build_reply_percept(
                current.text,
                current.requirement_type.value,
                current.nfr_category,
                current.confidence,
                current.justification,
                reason,
                self._lang,
            ),
            beliefs=[
                Belief(source="percept", content=current.text),
                Belief(source=f"message:{request.id}", content=reason),
            ],
            phase=EpisodePhase.REPLY,
        )

    @staticmethod
    def _report_of(item: ClassifiedRequirement) -> ClassificationReport:
        return ClassificationReport(
            text=item.text,
            text_en=item.text_en,
            requirement_type=item.requirement_type,
            nfr_category=item.nfr_category,
            confidence=item.confidence,
            justification=item.justification,
        )

    def _report_to_peer(self, item: ClassifiedRequirement) -> Message:
        return Message(
            sender=self.name,
            receiver=self._peer,
            performative=Performative.INFORM,
            requirement_id=item.id,
            content=self._report_of(item),
        )

    def _process_single(self, requirement: Requirement) -> ClassifiedRequirement:
        submission = self._run_episode(self._runtime, self._task_for(requirement))
        assert isinstance(submission.arguments, ClassificationSubmission)
        output = ClassificationOutput(
            requirement_id=requirement.id,
            requirement_type=submission.arguments.requirement_type,
            nfr_category=normalize_nfr_category(submission.arguments.nfr_category),
            confidence=submission.arguments.confidence,
            justification=submission.arguments.justification,
        )
        self._retain(requirement, output)
        return ClassifiedRequirement.from_requirement(requirement, output)

    def _task_for(self, requirement: Requirement) -> EpisodeTask:
        return EpisodeTask(
            agent=self.name,
            model=self.model,
            requirement_id=requirement.id,
            goal=self._goal,
            system_prompt=self._system_prompt,
            percept=build_classifier_percept(requirement.text, self._lang),
            beliefs=[Belief(source="percept", content=requirement.text)],
        )

    def _retain(self, requirement: Requirement, output: ClassificationOutput) -> None:
        """Memory hygiene, not control flow: only answers that are confident and consistent
        become examples for later items, so an unreliable answer does not propagate itself."""
        if self._memory is None or output.confidence < CONFIDENCE_THRESHOLD:
            return
        if structural_issues(output.requirement_type, output.nfr_category, self._valid_codes):
            return
        self._memory.add(
            requirement.id,
            requirement.text,
            output_summary=f"{output.requirement_type.value}/{output.nfr_category or '-'}",
            justification=output.justification,
        )

    def _detection_context(
        self,
        item: Requirement,
        output: ClassifiedRequirement | None,
        stage: str,
        parsed_ok: bool,
    ) -> DetectionContext | None:
        return DetectionContext(
            requirement_id=item.id,
            stage=stage,
            requirement_text=item.text,
            parsed_ok=parsed_ok,
            confidence=output.confidence if output is not None else None,
            predicted_category=output.nfr_category if output is not None else None,
            known_categories=set(NFR_ONLY_CODES),
            justification=output.justification if output is not None else "",
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
