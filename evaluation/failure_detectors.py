"""Pluggable, versioned failure detectors for SQ3 (ADR-003).

Each detector inspects a parsed agent output (via DetectionContext)
and optionally emits a FailureRecord. The DetectorChain runs an
ordered set of detectors after each parse. v1 uses deterministic
heuristics so the baseline is reproducible; LLM-as-judge is a future
v2 detector.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from domain.failures import FailureMode, FailureRecord, FailureSeverity

DETECTOR_CHAIN_VERSION = "v1"

_WORD_RE = re.compile(r"[a-zà-ú0-9]+")


def _content_words(text: str, min_len: int = 4) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= min_len}


@dataclass
class DetectionContext:
    """All signals a detector may inspect for one processed requirement."""

    requirement_id: str
    stage: str
    requirement_text: str
    parsed_ok: bool
    confidence: float | None = None
    predicted_category: str | None = None
    known_categories: set[str] | None = None
    justification: str = ""
    confidence_threshold: float = 0.5


Detector = Callable[[DetectionContext], FailureRecord | None]


def detect_schema(ctx: DetectionContext) -> FailureRecord | None:
    if ctx.parsed_ok:
        return None
    return FailureRecord(
        requirement_id=ctx.requirement_id,
        stage=ctx.stage,
        mode=FailureMode.SCHEMA_INVALID,
        severity=FailureSeverity.FATAL,
        evidence="parser fell back: no valid structured output",
    )


def detect_category_hallucination(ctx: DetectionContext) -> FailureRecord | None:
    if not ctx.known_categories or ctx.predicted_category is None:
        return None
    if ctx.predicted_category not in ctx.known_categories:
        return FailureRecord(
            requirement_id=ctx.requirement_id,
            stage=ctx.stage,
            mode=FailureMode.CATEGORY_HALLUCINATED,
            severity=FailureSeverity.DEGRADED,
            evidence=f"category '{ctx.predicted_category}' not in taxonomy",
        )
    return None


def detect_low_confidence(ctx: DetectionContext) -> FailureRecord | None:
    if ctx.confidence is None:
        return None
    if ctx.confidence < ctx.confidence_threshold:
        return FailureRecord(
            requirement_id=ctx.requirement_id,
            stage=ctx.stage,
            mode=FailureMode.LOW_CONFIDENCE,
            severity=FailureSeverity.FLAGGED,
            evidence=(
                f"confidence {ctx.confidence:.2f} < threshold {ctx.confidence_threshold:.2f}"
            ),
        )
    return None


def detect_grounding(ctx: DetectionContext) -> FailureRecord | None:
    """v1 lexical heuristic: justification with zero content-word overlap
    with the requirement text is likely ungrounded."""
    if not ctx.justification.strip():
        return None
    req_words = _content_words(ctx.requirement_text)
    just_words = _content_words(ctx.justification)
    if req_words and not (req_words & just_words):
        return FailureRecord(
            requirement_id=ctx.requirement_id,
            stage=ctx.stage,
            mode=FailureMode.GROUNDING_MISSING,
            severity=FailureSeverity.FLAGGED,
            evidence="no lexical overlap between justification and requirement",
        )
    return None


@dataclass
class DetectorChain:
    """Ordered, versioned set of detectors run after each parse."""

    detectors: list[Detector] = field(default_factory=list)
    version: str = DETECTOR_CHAIN_VERSION

    def run(self, ctx: DetectionContext) -> list[FailureRecord]:
        records: list[FailureRecord] = []
        for detector in self.detectors:
            record = detector(ctx)
            if record is not None:
                record.detector_version = self.version
                records.append(record)
        return records


def default_chain() -> DetectorChain:
    """The v1 deterministic chain used as the SQ3 baseline.

    Detectors are self-gating: each returns None when a required signal
    (confidence, predicted_category, ...) is absent from the
    DetectionContext, so the same chain can run uniformly across stages
    with different available signals (e.g. classification has confidence
    and category; prioritization has neither) without hand-picking a
    subset per stage.
    """
    return DetectorChain(
        detectors=[
            detect_schema,
            detect_category_hallucination,
            detect_low_confidence,
            detect_grounding,
        ]
    )


def max_severity(records: list[FailureRecord]) -> FailureSeverity | None:
    """Most severe record in the batch (fatal > degraded > flagged), or None."""
    if not records:
        return None
    order = {FailureSeverity.FATAL: 0, FailureSeverity.DEGRADED: 1, FailureSeverity.FLAGGED: 2}
    return min((r.severity for r in records), key=lambda s: order[s])
