"""Observe -> decide -> act loop for one agent working on one item.

The loop holds no procedure. It shows the model a percept, executes whatever action the model
emits, shows the observation back, and ends when the model submits its answer or the step budget
runs out. Which tools are called, in what order and how often is the model's choice; the code only
supplies the action space, validates actions against their schemas and enforces the budget
(ReAct, Yao et al., 2023; Russell & Norvig's sense-think-act cycle).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from agents.base import llm_retry
from domain.agent_actions import (
    ActionKind,
    ActionRecord,
    AgentEpisode,
    Belief,
    EpisodeOutcome,
    EpisodePhase,
    Goal,
)
from llm.json_parser import extract_first_json

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 5
_RECORDED = "Recorded."


@dataclass(frozen=True)
class EpisodeTask:
    """What the environment hands an agent: who it is, its goal, what it perceives."""

    agent: str
    model: str
    requirement_id: str
    goal: Goal
    system_prompt: str
    percept: str
    beliefs: list[Belief] = field(default_factory=list)
    phase: EpisodePhase = EpisodePhase.WORK


@dataclass(frozen=True)
class Submission:
    """The agent's final answer: the terminal action it chose and its validated arguments."""

    action: str
    arguments: BaseModel


@dataclass(frozen=True)
class EpisodeResult:
    episode: AgentEpisode
    submission: Submission | None


@dataclass
class _Step:
    records: list[ActionRecord] = field(default_factory=list)
    feedback: list[BaseMessage] = field(default_factory=list)
    beliefs: list[Belief] = field(default_factory=list)
    submission: Submission | None = None
    via: str | None = None


def _text_of(response: AIMessage) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


class AgentRuntime:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        terminal_tools: frozenset[str],
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._schemas = {tool.name: self._schema_of(tool) for tool in tools}
        missing = terminal_tools - self._tools.keys()
        if missing:
            raise ValueError(f"Terminal tools not among the agent's tools: {sorted(missing)}")
        self._terminal = terminal_tools
        self._max_steps = max_steps
        self._model = llm.bind_tools(list(tools))

    def description(self) -> dict[str, Any]:
        return {
            "max_steps": self._max_steps,
            "tools": sorted(self._tools),
            "terminal_tools": sorted(self._terminal),
        }

    @staticmethod
    def _schema_of(tool: BaseTool) -> type[BaseModel]:
        schema = tool.args_schema
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise ValueError(f"Tool {tool.name!r} needs a pydantic args_schema")
        return schema

    def run(self, task: EpisodeTask) -> EpisodeResult:
        episode = AgentEpisode(
            agent=task.agent,
            requirement_id=task.requirement_id,
            model=task.model,
            goal=task.goal,
            beliefs=list(task.beliefs),
            phase=task.phase,
        )
        messages: list[BaseMessage] = [
            SystemMessage(content=task.system_prompt),
            HumanMessage(content=task.percept),
        ]
        for step_number in range(1, self._max_steps + 1):
            response, latency_ms = self._decide(messages)
            messages.append(response)
            step = self._act(step_number, response)
            self._attach_call_cost(step, response, latency_ms)
            episode.actions.extend(step.records)
            episode.beliefs.extend(step.beliefs)
            if step.submission is not None:
                episode.outcome = EpisodeOutcome.SUBMITTED
                episode.submitted_via = "text" if step.via == "text" else "tool"
                return EpisodeResult(episode, step.submission)
            messages.extend(step.feedback)
        return EpisodeResult(episode, None)

    def _decide(self, messages: list[BaseMessage]) -> tuple[AIMessage, float]:
        started = time.monotonic()
        response = self._invoke(messages)
        return response, round((time.monotonic() - started) * 1000, 2)

    @llm_retry
    def _invoke(self, messages: list[BaseMessage]) -> AIMessage:
        response = self._model.invoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError(f"Model returned {type(response).__name__}, expected AIMessage")
        return response

    @staticmethod
    def _attach_call_cost(step: _Step, response: AIMessage, latency_ms: float) -> None:
        first = step.records[0]
        usage: Mapping[str, Any] = response.usage_metadata or {}
        first.input_tokens = usage.get("input_tokens", 0)
        first.output_tokens = usage.get("output_tokens", 0)
        first.latency_ms = latency_ms
        first.thought = _text_of(response)

    def _act(self, step_number: int, response: AIMessage) -> _Step:
        if not response.tool_calls:
            return self._act_on_text(step_number, response)
        step = _Step()
        for call in response.tool_calls:
            self._execute(step_number, call, step)
            if step.submission is not None:
                break
        return step

    def _execute(self, step_number: int, call: ToolCall, step: _Step) -> None:
        name = call["name"]
        arguments = call.get("args") or {}
        call_id = call.get("id") or f"call_{step_number}_{len(step.records)}"

        def reject(reason: str) -> None:
            self._record(step, step_number, ActionKind.INVALID, name, arguments, reason, call_id)

        if name not in self._tools:
            reject(f"Unknown tool {name!r}. Available: {', '.join(sorted(self._tools))}.")
            return
        try:
            validated = self._schemas[name].model_validate(arguments)
        except ValidationError as error:
            reject(f"Invalid arguments for {name!r}: {error.errors(include_url=False)}")
            return
        if name in self._terminal:
            self._record(step, step_number, ActionKind.SUBMIT, name, arguments, _RECORDED, call_id)
            step.submission = Submission(name, validated)
            return
        try:
            observation = str(self._tools[name].invoke(arguments))
        except Exception as error:
            reject(f"Tool {name!r} failed: {error}")
            return
        self._record(step, step_number, ActionKind.TOOL, name, arguments, observation, call_id)
        step.beliefs.append(Belief(source=f"tool:{name}", content=observation))

    @staticmethod
    def _record(
        step: _Step,
        step_number: int,
        kind: ActionKind,
        name: str,
        arguments: dict[str, Any],
        observation: str,
        call_id: str,
    ) -> None:
        step.records.append(
            ActionRecord(
                step=step_number,
                kind=kind,
                name=name,
                arguments=arguments,
                observation=observation,
            )
        )
        step.feedback.append(ToolMessage(content=observation, tool_call_id=call_id))

    def _act_on_text(self, step_number: int, response: AIMessage) -> _Step:
        """A reply with no tool call counts as a submission only if it is a valid answer in
        JSON; anything else is an invalid action and the model is told how to act."""
        step = _Step()
        text = _text_of(response)
        submission = self._submission_from_text(text)
        if submission is not None:
            step.submission, step.via = submission, "text"
            step.records.append(
                ActionRecord(
                    step=step_number,
                    kind=ActionKind.SUBMIT,
                    name=submission.action,
                    arguments=submission.arguments.model_dump(mode="json"),
                    observation=_RECORDED,
                )
            )
            return step
        reminder = (
            "No valid action was taken. Act by calling one of the tools "
            f"({', '.join(sorted(self._tools))}); finish by calling "
            f"{' or '.join(sorted(self._terminal))}."
        )
        step.records.append(
            ActionRecord(step=step_number, kind=ActionKind.INVALID, observation=reminder)
        )
        step.feedback.append(HumanMessage(content=reminder))
        return step

    def _submission_from_text(self, text: str) -> Submission | None:
        try:
            data = json.loads(extract_first_json(text))
        except Exception:
            return None
        for name in sorted(self._terminal):
            try:
                return Submission(name, self._schemas[name].model_validate(data))
            except ValidationError:
                continue
        return None
