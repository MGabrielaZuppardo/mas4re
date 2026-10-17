"""The mediator of protocol P3: reads the finished backlog, finds problems, asks for decisions
to be reconsidered.

It belongs to the environment (like the cross-check), not to the agents. Its only power is to send
a typed request with the evidence; the prioritizer decides whether to agree or refuse. Two kinds
of finding:

* conflict -- a critical NFR category (security, performance, availability, fault tolerance) with
  a Could/Won't-Have priority (evaluation/cross_agent_check.py);
* Must inflation -- more than 60% of the backlog as Must (the DSDM guideline), in which case the
  Musts with the lowest scores are put to the prioritizer.

A single round, and at most `max_requests` requests, so the cost and the dialogue stay bounded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from domain.enums import Lang, MoSCoWPriority
from domain.messages import Message, Performative, ReprioritizationRequest
from domain.models import PrioritizedRequirement
from evaluation.cross_agent_check import detect_inter_agent_conflict
from prompts.v2.mediation import conflict_evidence, inflation_evidence, mediator_reason

MEDIATOR = "mediator"
MUST_SHARE_LIMIT = 0.60
DEFAULT_MAX_REQUESTS = 25


class FindingKind(StrEnum):
    CONFLICT = "conflict"
    MUST_INFLATION = "must_inflation"


@dataclass(frozen=True)
class Finding:
    requirement_id: str
    kind: FindingKind
    evidence: str


@dataclass(frozen=True)
class MediationReport:
    findings: list[Finding]
    conflicts: int
    must_inflation: int
    capped: bool
    must_share: float


def must_share(items: Sequence[PrioritizedRequirement]) -> float:
    musts = sum(1 for item in items if item.priority is MoSCoWPriority.MUST_HAVE)
    return musts / len(items) if items else 0.0


def _score(item: PrioritizedRequirement) -> float:
    if item.priority_score is not None:
        return item.priority_score
    return item.priority.score if item.priority is not None else 0.0


def find_conflicts(items: Sequence[PrioritizedRequirement], lang: Lang) -> list[Finding]:
    return [
        Finding(
            item.id,
            FindingKind.CONFLICT,
            conflict_evidence(str(item.nfr_category), item.priority.value, lang),
        )
        for item in items
        if item.priority is not None and detect_inter_agent_conflict(item) is not None
    ]


def find_must_inflation(
    items: Sequence[PrioritizedRequirement], lang: Lang, limit: float = MUST_SHARE_LIMIT
) -> list[Finding]:
    """The weakest Musts, as many as it takes to bring the Must share back to the limit."""
    musts = [item for item in items if item.priority is MoSCoWPriority.MUST_HAVE]
    excess = len(musts) - math.floor(limit * len(items))
    if excess <= 0:
        return []
    share = len(musts) / len(items)
    weakest = sorted(musts, key=lambda item: (_score(item), -(item.priority_rank or 0)))[:excess]
    return [
        Finding(
            item.id,
            FindingKind.MUST_INFLATION,
            inflation_evidence(share, limit, _score(item), lang),
        )
        for item in weakest
    ]


def mediate(
    items: Sequence[PrioritizedRequirement],
    lang: Lang,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> MediationReport:
    conflicts = find_conflicts(items, lang)
    inflation = find_must_inflation(items, lang)
    first_finding_per_item: dict[str, Finding] = {}
    for finding in [*conflicts, *inflation]:
        first_finding_per_item.setdefault(finding.requirement_id, finding)
    chosen = list(first_finding_per_item.values())
    return MediationReport(
        findings=chosen[:max_requests],
        conflicts=len(conflicts),
        must_inflation=len(inflation),
        capped=len(chosen) > max_requests,
        must_share=must_share(items),
    )


def request_for(finding: Finding, lang: Lang, receiver: str = "prioritizer") -> Message:
    return Message(
        sender=MEDIATOR,
        receiver=receiver,
        performative=Performative.REQUEST,
        requirement_id=finding.requirement_id,
        content=ReprioritizationRequest(
            reason=mediator_reason(finding.kind.value, lang), evidence=finding.evidence
        ),
    )
