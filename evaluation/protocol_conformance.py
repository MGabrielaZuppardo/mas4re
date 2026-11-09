"""Did the run obey the architecture it claims to have run?

`check_conformance` reads the messages of a finished run and reports every one that the protocol's
spec does not allow, and every exchange that breaks its rules (a reply to nothing, a reply from
the wrong agent, more requests per requirement than the protocol permits).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from domain.messages import Message, Performative
from pipeline.protocol_spec import ProtocolSpec

_MAX_EXAMPLES = 20

OUTSIDE_CHANNEL = "outside_channel"
UNKNOWN_SENDER = "unknown_participant"
SELF_MESSAGE = "self_message"
REPLY_TO_NOTHING = "reply_to_nothing"
REPLY_FROM_WRONG_AGENT = "reply_from_wrong_agent"
TOO_MANY_REQUESTS = "too_many_requests"


@dataclass(frozen=True)
class Violation:
    rule: str
    message_id: str
    requirement_id: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "rule": self.rule,
            "message_id": self.message_id,
            "requirement_id": self.requirement_id,
            "detail": self.detail,
        }


def _check_message(spec: ProtocolSpec, message: Message) -> Violation | None:
    where = (message.id, message.requirement_id)
    if message.sender == message.receiver:
        return Violation(SELF_MESSAGE, *where, f"{message.sender} wrote to itself")
    if {message.sender, message.receiver} - set(spec.roles):
        return Violation(
            UNKNOWN_SENDER, *where, f"{message.sender}->{message.receiver} not in {spec.roles}"
        )
    kind = message.content.kind
    if not spec.allows(message.sender, message.receiver, message.performative, kind):
        return Violation(
            OUTSIDE_CHANNEL,
            *where,
            f"{message.sender}->{message.receiver} {message.performative.value}/{kind}",
        )
    return None


def _check_reply(message: Message, requests: dict[str, Message]) -> Violation | None:
    if message.in_reply_to is None or message.performative is Performative.INFORM:
        return None
    where = (message.id, message.requirement_id)
    request = requests.get(message.in_reply_to)
    if request is None:
        return Violation(REPLY_TO_NOTHING, *where, f"no request {message.in_reply_to}")
    if request.receiver != message.sender or request.requirement_id != message.requirement_id:
        return Violation(
            REPLY_FROM_WRONG_AGENT,
            *where,
            f"{message.sender} answered {request.receiver}'s request",
        )
    return None


def check_conformance(spec: ProtocolSpec, messages: Sequence[Message]) -> list[Violation]:
    requests = {m.id: m for m in messages if m.performative is Performative.REQUEST}
    violations = [v for m in messages if (v := _check_message(spec, m))]
    violations += [v for m in messages if (v := _check_reply(m, requests))]
    per_requirement = Counter((m.sender, m.requirement_id) for m in requests.values())
    for (sender, requirement_id), count in sorted(per_requirement.items()):
        if count > spec.max_requests_per_requirement:
            violations.append(
                Violation(
                    TOO_MANY_REQUESTS,
                    "",
                    requirement_id,
                    f"{sender} sent {count} requests, the protocol allows "
                    f"{spec.max_requests_per_requirement}",
                )
            )
    return violations


def summarize_conformance(spec: ProtocolSpec, messages: Sequence[Message]) -> dict[str, Any]:
    violations = check_conformance(spec, messages)
    return {
        "messages_checked": len(messages),
        "violations": len(violations),
        "by_rule": dict(sorted(Counter(v.rule for v in violations).items())),
        "examples": [v.as_dict() for v in violations[:_MAX_EXAMPLES]],
    }
