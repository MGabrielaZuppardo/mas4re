from __future__ import annotations

from domain.failures import FailureMode, FailureSeverity
from evaluation.failure_detectors import (
    DetectionContext,
    DetectorChain,
    default_chain,
    detect_category_hallucination,
    detect_grounding,
    detect_low_confidence,
    detect_schema,
)


def _ctx(**kw: object) -> DetectionContext:
    base: dict[str, object] = {
        "requirement_id": "req-01",
        "stage": "classifier",
        "requirement_text": "The system must authenticate users securely.",
        "parsed_ok": True,
    }
    base.update(kw)
    return DetectionContext(**base)  # type: ignore[arg-type]


class TestDetectSchema:
    def test_ok_returns_none(self) -> None:
        assert detect_schema(_ctx(parsed_ok=True)) is None

    def test_invalid_emits_fatal(self) -> None:
        rec = detect_schema(_ctx(parsed_ok=False))
        assert rec is not None
        assert rec.mode is FailureMode.SCHEMA_INVALID
        assert rec.severity is FailureSeverity.FATAL


class TestDetectHallucination:
    def test_unknown_category_flagged(self) -> None:
        rec = detect_category_hallucination(
            _ctx(predicted_category="ZZ", known_categories={"SE", "PE"})
        )
        assert rec is not None
        assert rec.mode is FailureMode.CATEGORY_HALLUCINATED

    def test_known_category_ok(self) -> None:
        assert (
            detect_category_hallucination(
                _ctx(predicted_category="SE", known_categories={"SE", "PE"})
            )
            is None
        )

    def test_no_taxonomy_skips(self) -> None:
        assert detect_category_hallucination(_ctx(predicted_category="anything")) is None


class TestDetectLowConfidence:
    def test_below_threshold(self) -> None:
        rec = detect_low_confidence(_ctx(confidence=0.2, confidence_threshold=0.5))
        assert rec is not None
        assert rec.mode is FailureMode.LOW_CONFIDENCE

    def test_above_threshold_ok(self) -> None:
        assert detect_low_confidence(_ctx(confidence=0.9)) is None

    def test_none_confidence_skips(self) -> None:
        assert detect_low_confidence(_ctx(confidence=None)) is None


class TestDetectGrounding:
    def test_no_overlap_flagged(self) -> None:
        rec = detect_grounding(
            _ctx(
                requirement_text="The system must encrypt stored passwords.",
                justification="Banana elephant rainbow xylophone.",
            )
        )
        assert rec is not None
        assert rec.mode is FailureMode.GROUNDING_MISSING

    def test_overlap_ok(self) -> None:
        assert (
            detect_grounding(
                _ctx(
                    requirement_text="The system must encrypt passwords.",
                    justification="Mentions encrypt and passwords security.",
                )
            )
            is None
        )

    def test_empty_justification_skips(self) -> None:
        assert detect_grounding(_ctx(justification="")) is None


class TestDetectorChain:
    def test_default_chain_has_four_detectors(self) -> None:
        chain = default_chain()
        assert len(chain.detectors) == 4
        assert chain.version == "v1"

    def test_chain_collects_multiple(self) -> None:
        chain = default_chain()
        ctx = _ctx(
            parsed_ok=False,
            confidence=0.1,
            confidence_threshold=0.5,
        )
        records = chain.run(ctx)
        modes = {r.mode for r in records}
        assert FailureMode.SCHEMA_INVALID in modes
        assert FailureMode.LOW_CONFIDENCE in modes
        assert all(r.detector_version == "v1" for r in records)

    def test_empty_chain_returns_empty(self) -> None:
        assert DetectorChain().run(_ctx()) == []
