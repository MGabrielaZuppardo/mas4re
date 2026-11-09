"""Failures of the exchange between agents, read from the messages of a finished run.

A protocol can fail without any single agent failing: a request goes unanswered, an agent agrees
and changes nothing, or refuses and changes it anyway. These are the inter-agent misalignment
failures that the MAS literature reports, so they are recorded apart from the per-item failures.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from domain.failures import FailureMode, FailureRecord, FailureSeverity
from domain.messages import Message, Performative

STAGE = "interaction"
_MAX_EXAMPLES = 20


def _record(
    message: Message, mode: FailureMode, severity: FailureSeverity, evidence: str
) -> FailureRecord:
    return FailureRecord(
        requirement_id=message.requirement_id,
        stage=STAGE,
        mode=mode,
        severity=severity,
        evidence=evidence,
    )


def _unanswered(requests: list[Message], replies: dict[str, Message]) -> list[FailureRecord]:
    found = []
    for request in requests:
        reply = replies.get(request.id)
        if reply is None:
            found.append(
                _record(
                    request,
                    FailureMode.INTERACTION_UNANSWERED,
                    FailureSeverity.DEGRADED,
                    f"request {request.id} from {request.sender} got no reply",
                )
            )
        elif reply.performative is Performative.FAILURE:
            found.append(
                _record(
                    reply,
                    FailureMode.INTERACTION_UNANSWERED,
                    FailureSeverity.DEGRADED,
                    f"{reply.sender} could not answer request {request.id}",
                )
            )
    return found


def _answer_ignored(
    messages: Sequence[Message], replies: dict[str, Message]
) -> list[FailureRecord]:
    acted_on = {m.in_reply_to for m in messages if m.performative is Performative.INFORM}
    return [
        _record(
            reply,
            FailureMode.INTERACTION_UNANSWERED,
            FailureSeverity.FLAGGED,
            f"{reply.receiver} did not act on the answer {reply.id}",
        )
        for reply in replies.values()
        if reply.performative in (Performative.AGREE, Performative.REFUSE)
        and reply.content.kind == "request_reply"
        and _is_reclassification(reply, messages)
        and reply.id not in acted_on
    ]


def _is_reclassification(reply: Message, messages: Sequence[Message]) -> bool:
    request = next((m for m in messages if m.id == reply.in_reply_to), None)
    return request is not None and request.content.kind == "reclassification_request"


def _contradicted(
    requests: list[Message], replies: dict[str, Message], decisions: dict[str, dict[str, Any]]
) -> list[FailureRecord]:
    found = []
    for request in requests:
        reply = replies.get(request.id)
        if reply is None or request.content.kind != "reprioritization_request":
            continue
        row = decisions.get(request.requirement_id)
        if row is None:
            continue
        changed = row["initial_priority"] != row["final_priority"]
        if reply.performative is Performative.AGREE and not changed:
            evidence = f"{reply.sender} agreed to revise but the priority did not change"
        elif reply.performative is Performative.REFUSE and changed:
            evidence = f"{reply.sender} refused but the priority changed"
        else:
            continue
        found.append(
            _record(reply, FailureMode.INTERACTION_INCONSISTENT, FailureSeverity.FLAGGED, evidence)
        )
    return found


def detect_interaction_failures(
    messages: Sequence[Message], decisions: Sequence[dict[str, Any]] = ()
) -> list[FailureRecord]:
    requests = [m for m in messages if m.performative is Performative.REQUEST]
    replies = {
        m.in_reply_to: m
        for m in messages
        if m.in_reply_to is not None and m.performative is not Performative.INFORM
    }
    by_id = {row["id"]: row for row in decisions}
    return [
        *_unanswered(requests, replies),
        *_answer_ignored(messages, {r.id: r for r in replies.values()}),
        *_contradicted(requests, replies, by_id),
    ]


def summarize_interaction_failures(
    messages: Sequence[Message], decisions: Sequence[dict[str, Any]] = ()
) -> dict[str, Any]:
    failures = detect_interaction_failures(messages, decisions)
    return {
        "total": len(failures),
        "by_mode": dict(sorted(Counter(f.mode.value for f in failures).items())),
        "examples": [
            {"requirement_id": f.requirement_id, "mode": f.mode.value, "evidence": f.evidence}
            for f in failures[:_MAX_EXAMPLES]
        ],
    }
