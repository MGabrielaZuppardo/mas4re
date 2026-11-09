"""The architecture of each coordination protocol, as data.

Who may send what to whom is what separates P0, P2 and P3, so it is written here once, instead of
being implied by graph nodes and tool lists. The graphs (`pipeline/protocols.py`) build it, and
`evaluation/protocol_conformance.py` checks a finished run against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from domain.enums import CoordinationProtocol
from domain.messages import Performative
from pipeline.mediation import MEDIATOR

CLASSIFIER = "classifier"
PRIORITIZER = "prioritizer"

_ANSWERS = frozenset({Performative.AGREE, Performative.REFUSE, Performative.FAILURE})


@dataclass(frozen=True)
class Channel:
    """One direction of communication: who may send which acts, carrying which content."""

    sender: str
    receiver: str
    performatives: frozenset[Performative]
    content_kinds: frozenset[str]

    def allows(self, sender: str, receiver: str, performative: Performative, kind: str) -> bool:
        return (
            sender == self.sender
            and receiver == self.receiver
            and performative in self.performatives
            and kind in self.content_kinds
        )


@dataclass(frozen=True)
class ProtocolSpec:
    protocol: CoordinationProtocol
    roles: tuple[str, ...]
    channels: tuple[Channel, ...]
    max_requests_per_requirement: int = 1

    def allows(self, sender: str, receiver: str, performative: Performative, kind: str) -> bool:
        return any(c.allows(sender, receiver, performative, kind) for c in self.channels)

    def describe(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.value,
            "roles": list(self.roles),
            "channels": [
                {
                    "from": c.sender,
                    "to": c.receiver,
                    "performatives": sorted(p.value for p in c.performatives),
                    "content": sorted(c.content_kinds),
                }
                for c in self.channels
            ],
            "max_requests_per_requirement": self.max_requests_per_requirement,
        }


_REPORT = Channel(
    CLASSIFIER, PRIORITIZER, frozenset({Performative.INFORM}), frozenset({"classification_report"})
)
_ASK_CLASSIFIER = Channel(
    PRIORITIZER,
    CLASSIFIER,
    frozenset({Performative.REQUEST}),
    frozenset({"reclassification_request"}),
)
_CLASSIFIER_ANSWERS = Channel(CLASSIFIER, PRIORITIZER, _ANSWERS, frozenset({"request_reply"}))
_PRIORITY_REPORT = Channel(
    PRIORITIZER, CLASSIFIER, frozenset({Performative.INFORM}), frozenset({"priority_report"})
)
_MEDIATOR_ASKS = Channel(
    MEDIATOR,
    PRIORITIZER,
    frozenset({Performative.REQUEST}),
    frozenset({"reprioritization_request"}),
)
_PRIORITIZER_ANSWERS = Channel(PRIORITIZER, MEDIATOR, _ANSWERS, frozenset({"request_reply"}))

PROTOCOL_SPECS: dict[CoordinationProtocol, ProtocolSpec] = {
    CoordinationProtocol.FIXED: ProtocolSpec(
        CoordinationProtocol.FIXED, (CLASSIFIER, PRIORITIZER), (_REPORT,)
    ),
    CoordinationProtocol.PEERS: ProtocolSpec(
        CoordinationProtocol.PEERS,
        (CLASSIFIER, PRIORITIZER),
        (_REPORT, _ASK_CLASSIFIER, _CLASSIFIER_ANSWERS, _PRIORITY_REPORT),
    ),
    CoordinationProtocol.MEDIATED: ProtocolSpec(
        CoordinationProtocol.MEDIATED,
        (CLASSIFIER, PRIORITIZER, MEDIATOR),
        (_REPORT, _MEDIATOR_ASKS, _PRIORITIZER_ANSWERS),
    ),
}
