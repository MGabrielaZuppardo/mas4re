"""Failure taxonomy for SQ3 (ADR-003).

Separates genuine model failures from infrastructure noise so that
SQ3 can answer questions like "does the pipeline hallucinate more
than the baseline?" without contaminating quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class FailureMode(StrEnum):
    """Categorized failure modes observed in LLM-based pipelines."""

    INFRA_TRANSIENT = "infra_transient"
    SCHEMA_INVALID = "schema_invalid"
    CATEGORY_HALLUCINATED = "category_hallucinated"
    LOW_CONFIDENCE = "low_confidence"
    GROUNDING_MISSING = "grounding_missing"
    AMBIGUOUS = "ambiguous"
    INTER_AGENT_CONFLICT = "inter_agent_conflict"


class FailureSeverity(StrEnum):
    """How a failure affects the run.

    fatal:    output unusable (excluded from quality metrics).
    degraded: output usable but suspect (kept, flagged).
    flagged:  output plausible but worth auditing.
    """

    FATAL = "fatal"
    DEGRADED = "degraded"
    FLAGGED = "flagged"


class FailureRecord(BaseModel):
    """A single categorized failure, emitted by a detector."""

    requirement_id: str
    stage: str
    mode: FailureMode
    severity: FailureSeverity
    evidence: str = Field(default="")
    detector_version: str = Field(default="v1")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass
class BatchResult(Generic[T]):
    """Partitions a processed batch into successes, failures and flagged.

    Quality metrics are computed over ``successes`` + ``flagged``;
    failure metrics are computed over ``failures`` (ADR-003).
    """

    successes: list[T] = field(default_factory=list)
    failures: list[FailureRecord] = field(default_factory=list)
    flagged: list[T] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.successes) + len(self.failures) + len(self.flagged)

    @property
    def failure_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return round(len(self.failures) / self.n_total, 4)

    @property
    def flagged_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return round(len(self.flagged) / self.n_total, 4)
