"""Fault-injection tests for the typed-state validation boundary (§6.6).

Confirms, for more than one synthetic field-constraint violation, that the
Pydantic-validated pipeline rejects an out-of-range field immediately after
the classification call, while the untyped two-call ablation propagates the
same value unchecked until final output assembly — where it either raises
late (uncaught by any parser-level try/except) or, if never re-validated,
is silently accepted.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.two_call_baseline import TwoCallBaselineAgent, _ClassificationResult
from domain.enums import MoSCoWPriority, RequirementType
from domain.models import BaselineOutput, ClassificationOutput


@pytest.fixture
def agent():
    from unittest.mock import patch

    with patch("agents.two_call_baseline.build_llm"):
        return TwoCallBaselineAgent(model="ollama/qwen2.5:7b")


class TestTypedPipelineRejectsImmediately:
    """ClassificationOutput (the pipeline's typed boundary) must reject an
    out-of-range field at construction time, before it ever reaches
    PipelineState or the prioritizer."""

    @pytest.mark.parametrize("bad_confidence", [1.4, -0.5, 2.0])
    def test_confidence_out_of_range_rejected(self, bad_confidence):
        with pytest.raises(ValidationError):
            ClassificationOutput(
                requirement_id="req-01",
                requirement_type=RequirementType.NON_FUNCTIONAL,
                nfr_category="SE",
                confidence=bad_confidence,
                justification="fault-injection probe",
            )

    @pytest.mark.parametrize("bad_priority_score", [1.5, -0.1])
    def test_priority_score_out_of_range_rejected(self, bad_priority_score):
        with pytest.raises(ValidationError):
            BaselineOutput(
                requirement_id="req-01",
                requirement_type=RequirementType.NON_FUNCTIONAL,
                nfr_category="SE",
                confidence=0.9,
                priority=MoSCoWPriority.MUST_HAVE,
                priority_score=bad_priority_score,
                priority_rank=1,
            )


class TestUntypedAblationPropagatesUnchecked:
    """The two-call ablation's intermediate _ClassificationResult is a plain
    dataclass: the same out-of-range values pass through _parse_classification
    unchecked, and only fail once BaselineOutput is assembled at the very end
    of _process_single — one LLM call later than the typed pipeline."""

    @pytest.mark.parametrize("bad_confidence", [1.4, -0.5, 2.0])
    def test_parse_classification_does_not_validate_confidence(self, agent, bad_confidence):
        content = (
            f'{{"requirement_type": "NF", "nfr_category": "SE", '
            f'"confidence": {bad_confidence}, "justification": "fault-injection probe"}}'
        )
        result = agent._parse_classification(content, "req-01")

        assert isinstance(result, _ClassificationResult)
        assert result.confidence == bad_confidence  # not clamped, not rejected

    @pytest.mark.parametrize("bad_confidence", [1.4, -0.5, 2.0])
    def test_final_assembly_rejects_the_same_value_one_call_later(self, bad_confidence):
        with pytest.raises(ValidationError):
            BaselineOutput(
                requirement_id="req-01",
                requirement_type=RequirementType.NON_FUNCTIONAL,
                nfr_category="SE",
                confidence=bad_confidence,  # propagated unchecked from Call 1
                priority=MoSCoWPriority.MUST_HAVE,
                priority_score=0.9,
                priority_rank=1,
            )


class TestUntypedAblationSilentlyDrops:
    """agents.two_call_baseline.TwoCallBaselineAgent._process_single wraps
    the whole two-call sequence in @retry with reraise=True; a
    ValidationError raised at final assembly propagates out of
    _process_single uncaught by any parser-level try/except, matching the
    paper's description of a silent drop with no failure record."""

    def test_process_single_reraises_validation_error(self, agent, monkeypatch):
        from domain.models import Requirement

        requirement = Requirement(id="req-01", text="O sistema deve ser seguro.")

        monkeypatch.setattr(
            agent,
            "_parse_classification",
            lambda content, req_id: _ClassificationResult(
                requirement_type=RequirementType.NON_FUNCTIONAL,
                nfr_category="SE",
                confidence=1.4,  # out-of-range, unchecked at this point
                justification="fault-injection probe",
            ),
        )

        class _FakeLLM:
            def invoke(self, messages):
                from unittest.mock import MagicMock

                mock = MagicMock()
                mock.content = (
                    '{"priority": "M", "priority_score": 0.9, "priority_rank": 1,'
                    ' "justification": "n/a"}'
                )
                return mock

        agent._llm = _FakeLLM()

        with pytest.raises(ValidationError):
            agent._process_single.__wrapped__(agent, requirement)
