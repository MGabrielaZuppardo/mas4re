"""Unit tests for domain enums and models — refactor/typed-core."""

from __future__ import annotations

import pytest

from domain.enums import Lang, MoSCoWPriority, RequirementType
from domain.models import (
    BaselineOutput,
    ClassifiedRequirement,
    PrioritizationOutput,
    PrioritizedRequirement,
    Requirement,
)

# ── Lang enum ───────────────────────────────────────────────────────────────

class TestLangEnum:
    def test_values(self) -> None:
        assert Lang.PT.value == "pt"
        assert Lang.EN.value == "en"

    def test_string_comparison(self) -> None:
        # StrEnum allows comparison with raw strings
        assert Lang.PT == "pt"
        assert Lang.EN == "en"

    def test_identity_comparison(self) -> None:
        # Internal recommended comparison
        lang = Lang.PT
        assert lang is Lang.PT
        assert lang is not Lang.EN

    def test_membership(self) -> None:
        assert Lang.PT in Lang
        assert Lang.EN in Lang

    def test_iteration(self) -> None:
        all_langs = list(Lang)
        assert len(all_langs) == 2
        assert Lang.PT in all_langs
        assert Lang.EN in all_langs

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            Lang("fr")


# ── PrioritizedRequirement renamed field ────────────────────────────────────

class TestPrioritizedRequirementRenamedField:
    """Garante que justification_priority -> priority_justification foi aplicado."""

    def _make_classified(self) -> ClassifiedRequirement:
        return ClassifiedRequirement(
            text="The system must authenticate users via OAuth.",
            requirement_type=RequirementType.NON_FUNCTIONAL,
            nfr_category="SE",
            confidence=0.9,
            justification="Mentions authentication mechanism.",
        )

    def test_field_exists_with_new_name(self) -> None:
        req = PrioritizedRequirement(
            text="x" * 10,
            requirement_type=RequirementType.FUNCTIONAL,
            priority=MoSCoWPriority.MUST_HAVE,
            priority_score=1.0,
            priority_rank=1,
            priority_justification="critical for auth flow",
        )
        assert req.priority_justification == "critical for auth flow"

    def test_old_field_name_no_longer_exists(self) -> None:
        # Pydantic v2: extra fields are ignored by default; passing the
        # old name should NOT populate the new field.
        req = PrioritizedRequirement(
            text="x" * 10,
            requirement_type=RequirementType.FUNCTIONAL,
            priority=MoSCoWPriority.MUST_HAVE,
            priority_score=1.0,
            priority_rank=1,
            justification_priority="legacy text",  # type: ignore[call-arg]
        )
        assert req.priority_justification == ""

    def test_from_classified_populates_priority_justification(self) -> None:
        classified = self._make_classified()
        output = PrioritizationOutput(
            requirement_id=classified.id,
            priority=MoSCoWPriority.SHOULD_HAVE,
            priority_score=0.75,
            priority_rank=2,
            justification="auth is non-trivial but not blocker",
        )
        prioritized = PrioritizedRequirement.from_classified(classified, output)
        assert prioritized.priority_justification == "auth is non-trivial but not blocker"

    def test_from_baseline_populates_priority_justification(self) -> None:
        req = Requirement(text="Users must be able to reset passwords.")
        baseline_out = BaselineOutput(
            requirement_id=req.id,
            requirement_type=RequirementType.FUNCTIONAL,
            nfr_category=None,
            confidence=0.85,
            classification_justification="describes user action",
            priority=MoSCoWPriority.MUST_HAVE,
            priority_score=1.0,
            priority_rank=1,
            priority_justification="critical recovery path",
        )
        prioritized = PrioritizedRequirement.from_baseline(req, baseline_out)
        assert prioritized.priority_justification == "critical recovery path"