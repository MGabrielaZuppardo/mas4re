"""Record of what an agent perceived, believed and did while working on one item.

One `AgentEpisode` per item and agent. Its actions are the sequence the model itself chose, so the
log doubles as the evidence that the next step is not dictated by code and as training data
(observation, action, outcome).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionKind(StrEnum):
    TOOL = "tool"
    SUBMIT = "submit"
    INVALID = "invalid"


class EpisodeOutcome(StrEnum):
    SUBMITTED = "submitted"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Goal(BaseModel):
    description: str


class Belief(BaseModel):
    """A fact the agent holds, and where it came from (percept, a tool, a message)."""

    source: str
    content: str


class ActionRecord(BaseModel):
    """One action the model emitted. Tokens and latency belong to the model call that produced
    it, and are attached to the first action of that call only."""

    step: int = Field(ge=1)
    kind: ActionKind
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    thought: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class AgentEpisode(BaseModel):
    agent: str
    requirement_id: str
    model: str
    goal: Goal
    beliefs: list[Belief] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)
    outcome: EpisodeOutcome = EpisodeOutcome.BUDGET_EXHAUSTED
    submitted_via: Literal["tool", "text"] | None = None

    @property
    def llm_calls(self) -> int:
        return len({a.step for a in self.actions})

    @property
    def input_tokens(self) -> int:
        return sum(a.input_tokens for a in self.actions)

    @property
    def output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.actions)

    @property
    def action_sequence(self) -> tuple[str, ...]:
        """Names of the valid actions in the order the agent chose them."""
        return tuple(a.name for a in self.actions if a.kind is not ActionKind.INVALID)
