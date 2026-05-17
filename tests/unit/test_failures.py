"""Unit tests for the SQ3 failure taxonomy (ADR-003)."""

from __future__ import annotations

from domain.failures import (
    BatchResult,
    FailureMode,
    FailureRecord,
    FailureSeverity,
)


class TestFailureMode:
    def test_all_modes_present(self) -> None:
        expected = {
            "infra_transient",
            "schema_invalid",
            "category_hallucinated",
            "low_confidence",
            "grounding_missing",
            "ambiguous",
            "inter_agent_conflict",
        }
        assert {m.value for m in FailureMode} == expected

    def test_string_comparison(self) -> None:
        assert FailureMode.SCHEMA_INVALID == "schema_invalid"


class TestFailureRecord:
    def test_minimal_record(self) -> None:
        rec = FailureRecord(
            requirement_id="req-01",
            stage="classifier",
            mode=FailureMode.CATEGORY_HALLUCINATED,
            severity=FailureSeverity.DEGRADED,
        )
        assert rec.requirement_id == "req-01"
        assert rec.mode is FailureMode.CATEGORY_HALLUCINATED
        assert rec.severity is FailureSeverity.DEGRADED
        assert rec.detector_version == "v1"
        assert rec.evidence == ""

    def test_serializable(self) -> None:
        rec = FailureRecord(
            requirement_id="r1",
            stage="prioritizer",
            mode=FailureMode.SCHEMA_INVALID,
            severity=FailureSeverity.FATAL,
            evidence="invalid JSON",
        )
        data = rec.model_dump(mode="json")
        assert data["mode"] == "schema_invalid"
        assert data["severity"] == "fatal"
        assert "timestamp" in data


class TestBatchResult:
    def test_empty(self) -> None:
        br: BatchResult[str] = BatchResult()
        assert br.n_total == 0
        assert br.failure_rate == 0.0
        assert br.flagged_rate == 0.0

    def test_rates(self) -> None:
        failure = FailureRecord(
            requirement_id="r1",
            stage="classifier",
            mode=FailureMode.LOW_CONFIDENCE,
            severity=FailureSeverity.FLAGGED,
        )
        br: BatchResult[str] = BatchResult(
            successes=["a", "b", "c", "d", "e", "f", "g", "h"],
            failures=[failure],
            flagged=["x"],
        )
        assert br.n_total == 10
        assert br.failure_rate == 0.1
        assert br.flagged_rate == 0.1
