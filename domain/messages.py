"""Typed messages exchanged between agents (FIPA-ACL style).

Agents never read each other's outputs directly: what one agent knows about the other's work
arrives as a `Message` in its inbox, with a performative that says what the sender is doing
(informing, requesting) and a typed payload that says what it is about.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from domain.enums import RequirementType


class Performative(StrEnum):
    REQUEST = "request"
    INFORM = "inform"
    AGREE = "agree"
    REFUSE = "refuse"
    FAILURE = "failure"


class ClassificationReport(BaseModel):
    """What a classifier tells the prioritizer about one requirement."""

    kind: Literal["classification_report"] = "classification_report"
    text: str
    text_en: str | None = None
    requirement_type: RequirementType
    nfr_category: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str = ""


class ReclassificationRequest(BaseModel):
    """A peer asks for a requirement to be classified again, with the reason why."""

    kind: Literal["reclassification_request"] = "reclassification_request"
    reason: str


class RequestReply(BaseModel):
    """Answer to a request: the reason, plus the revised report when the sender agreed."""

    kind: Literal["request_reply"] = "request_reply"
    reason: str
    revised: ClassificationReport | None = None


MessageContent = Annotated[
    ClassificationReport | ReclassificationRequest | RequestReply,
    Field(discriminator="kind"),
]


class Message(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    sender: str
    receiver: str
    performative: Performative
    requirement_id: str
    content: MessageContent
    in_reply_to: str | None = None


def inbox_of(
    messages: Iterable[Message],
    receiver: str,
    performative: Performative | None = None,
) -> list[Message]:
    return [
        m
        for m in messages
        if m.receiver == receiver and (performative is None or m.performative == performative)
    ]


def latest_per_requirement(messages: Iterable[Message]) -> list[Message]:
    """One message per requirement, the most recent, in order of first appearance.

    A requirement sent again (a second pass, a revision) replaces its earlier message.
    """
    latest: dict[str, Message] = {}
    for message in messages:
        latest[message.requirement_id] = message
    return list(latest.values())
