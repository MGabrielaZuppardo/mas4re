"""Inter-agent conflict detection for SQ3 (ADR-003).

This signal only exists in the multi-agent pipeline: it captures the
trade-off of adding coordination between specialized agents. A
conflict is an internal inconsistency between what the classifier
asserted (type / NFR category) and how the prioritizer ranked it.

v1 is deterministic and lexical so the baseline is reproducible
(LLM-as-judge is a future v2 detector).
"""

from __future__ import annotations

from domain.enums import MoSCoWPriority
from domain.failures import FailureMode, FailureRecord, FailureSeverity
from domain.models import PrioritizedRequirement

# NFR categories whose operational risk normally implies high priority
# (the prioritization prompt itself states SE/PE/A tend to be M or S).
_CRITICAL_NFR = {"SE", "PE", "A", "FT"}
_LOW_PRIORITIES = {MoSCoWPriority.COULD_HAVE, MoSCoWPriority.WONT_HAVE}

_STAGE = "cross_check"


def detect_inter_agent_conflict(
    req: PrioritizedRequirement,
) -> FailureRecord | None:
    """Flag a critical NFR (asserted by the classifier) that the
    prioritizer downgraded to Could/Won't-have."""
    if req.priority is None or req.nfr_category is None:
        return None
    if req.nfr_category in _CRITICAL_NFR and req.priority in _LOW_PRIORITIES:
        return FailureRecord(
            requirement_id=req.id,
            stage=_STAGE,
            mode=FailureMode.INTER_AGENT_CONFLICT,
            severity=FailureSeverity.DEGRADED,
            evidence=(
                f"classifier labelled critical NFR '{req.nfr_category}' "
                f"but prioritizer assigned '{req.priority.value}'"
            ),
        )
    return None


def check_requirements(
    requirements: list[PrioritizedRequirement],
) -> list[FailureRecord]:
    """Run the inter-agent conflict detector over a batch."""
    records: list[FailureRecord] = []
    for req in requirements:
        record = detect_inter_agent_conflict(req)
        if record is not None:
            records.append(record)
    return records
