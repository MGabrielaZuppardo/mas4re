from __future__ import annotations

from domain.enums import MoSCoWPriority, RequirementType
from domain.failures import FailureMode, FailureSeverity
from domain.models import PipelineState, PrioritizedRequirement
from evaluation.cross_agent_check import (
    check_requirements,
    detect_inter_agent_conflict,
)
from pipeline.nodes.cross_check_node import cross_check_node


def _req(category: str | None, priority: MoSCoWPriority) -> PrioritizedRequirement:
    return PrioritizedRequirement(
        text="The system must keep credentials encrypted at rest.",
        requirement_type=RequirementType.NON_FUNCTIONAL,
        nfr_category=category,
        priority=priority,
        priority_score=priority.score,
        priority_rank=1,
        priority_justification="t",
    )


class TestDetectInterAgentConflict:
    def test_critical_nfr_downgraded_is_conflict(self) -> None:
        rec = detect_inter_agent_conflict(_req("SE", MoSCoWPriority.WONT_HAVE))
        assert rec is not None
        assert rec.mode is FailureMode.INTER_AGENT_CONFLICT
        assert rec.severity is FailureSeverity.DEGRADED
        assert rec.stage == "cross_check"

    def test_critical_nfr_could_have_is_conflict(self) -> None:
        rec = detect_inter_agent_conflict(_req("PE", MoSCoWPriority.COULD_HAVE))
        assert rec is not None

    def test_critical_nfr_must_have_no_conflict(self) -> None:
        assert detect_inter_agent_conflict(_req("SE", MoSCoWPriority.MUST_HAVE)) is None

    def test_non_critical_nfr_low_priority_ok(self) -> None:
        assert detect_inter_agent_conflict(_req("US", MoSCoWPriority.WONT_HAVE)) is None

    def test_no_category_skips(self) -> None:
        assert detect_inter_agent_conflict(_req(None, MoSCoWPriority.WONT_HAVE)) is None


class TestCheckRequirements:
    def test_collects_only_conflicts(self) -> None:
        reqs = [
            _req("SE", MoSCoWPriority.WONT_HAVE),  # conflict
            _req("SE", MoSCoWPriority.MUST_HAVE),  # ok
            _req("FT", MoSCoWPriority.COULD_HAVE),  # conflict
        ]
        records = check_requirements(reqs)
        assert len(records) == 2
        assert all(r.mode is FailureMode.INTER_AGENT_CONFLICT for r in records)


class TestCrossCheckNode:
    def test_node_writes_conflicts_to_metrics(self) -> None:
        state = PipelineState(
            prioritized_requirements=[
                _req("SE", MoSCoWPriority.WONT_HAVE),
                _req("US", MoSCoWPriority.COULD_HAVE),
            ]
        )
        out = cross_check_node(state)
        conflicts = out.metrics["inter_agent_conflicts"]
        assert len(conflicts) == 1
        assert conflicts[0]["mode"] == "inter_agent_conflict"

    def test_node_empty_when_no_conflicts(self) -> None:
        state = PipelineState(prioritized_requirements=[_req("US", MoSCoWPriority.SHOULD_HAVE)])
        out = cross_check_node(state)
        assert out.metrics["inter_agent_conflicts"] == []
