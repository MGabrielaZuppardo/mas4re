"""Unit tests for datasets/translation_validator.py.

All tests are fully mocked — no Ollama, no BERTScore download required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from datasets.translation_validator import TranslationValidator, ValidationReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_csv(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    """Create a minimal PROMISE-PT CSV for testing."""
    if rows is None:
        rows = [
            {
                "ProjectID": "P1",
                "RequirementText": "The system shall process transactions.",
                "RequirementText_PT": "O sistema deve processar transações.",
                "class": "F",
            },
            {
                "ProjectID": "P1",
                "RequirementText": "The system shall respond within 2 seconds.",
                "RequirementText_PT": "O sistema deve responder em até 2 segundos.",
                "class": "PE",
            },
            {
                "ProjectID": "P2",
                "RequirementText": "The system shall encrypt all data.",
                "RequirementText_PT": "O sistema deve criptografar todos os dados.",
                "class": "SE",
            },
        ]
    path = tmp_path / "promise_pt_test.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def _mock_validator(
    model: str = "ollama/stub",
    back_texts: list[str] | None = None,
    f1_scores: list[float] | None = None,
) -> TranslationValidator:
    """Build a TranslationValidator with mocked LLM and BERTScore."""
    with patch("datasets.translation_validator.TranslationValidator._build_llm"):
        v = TranslationValidator(model=model)

    # Mock back_translate
    _back = back_texts or [
        "The system shall process transactions.",
        "The system shall respond within 2 seconds.",
        "The system shall encrypt all data.",
    ]
    v.back_translate = MagicMock(return_value=_back)

    # Mock compute_bertscore
    _f1 = f1_scores or [0.95, 0.88, 0.72]
    _p = [f + 0.01 for f in _f1]
    _r = [f - 0.01 for f in _f1]
    v.compute_bertscore = MagicMock(return_value=(_p, _r, _f1))

    return v


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    def _make_report(self, f1_scores: list[float]) -> ValidationReport:
        report = ValidationReport()
        for i, f1 in enumerate(f1_scores):
            report.rows.append(
                {
                    "index": i,
                    "project_id": "P1",
                    "class": "F",
                    "text_en_original": f"req {i}",
                    "text_pt_translated": f"req {i} pt",
                    "text_en_back": f"req {i} back",
                    "bertscore_precision": f1 + 0.01,
                    "bertscore_recall": f1 - 0.01,
                    "bertscore_f1": f1,
                    "decision": (
                        "accepted" if f1 >= 0.90 else "review" if f1 >= 0.80 else "rejected"
                    ),
                }
            )
        return report

    def test_n_total(self) -> None:
        report = self._make_report([0.95, 0.85, 0.70])
        assert report.n_total == 3

    def test_n_accepted(self) -> None:
        report = self._make_report([0.95, 0.91, 0.85, 0.70])
        assert report.n_accepted == 2

    def test_n_review(self) -> None:
        report = self._make_report([0.95, 0.85, 0.82, 0.70])
        assert report.n_review == 2

    def test_n_rejected(self) -> None:
        report = self._make_report([0.95, 0.85, 0.70, 0.65])
        assert report.n_rejected == 2

    def test_mean_f1(self) -> None:
        report = self._make_report([0.90, 0.80, 0.70])
        assert report.mean_f1 == pytest.approx(0.80, abs=1e-9)

    def test_summary_keys(self) -> None:
        report = self._make_report([0.95, 0.85, 0.70])
        s = report.summary()
        assert set(s.keys()) == {
            "n_total",
            "n_accepted",
            "n_review",
            "n_rejected",
            "mean_bertscore_f1",
            "threshold_accept",
            "threshold_review",
        }

    def test_save_creates_csv(self, tmp_path: Path) -> None:
        report = self._make_report([0.95, 0.85, 0.70])
        out = tmp_path / "report.csv"
        report.save(out)
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 3
        assert "bertscore_f1" in df.columns
        assert "decision" in df.columns

    def test_df_has_correct_columns(self) -> None:
        report = self._make_report([0.95])
        cols = set(report.df.columns)
        assert {"bertscore_f1", "bertscore_precision", "bertscore_recall", "decision"} <= cols


# ---------------------------------------------------------------------------
# TranslationValidator._decide
# ---------------------------------------------------------------------------


class TestDecide:
    def setup_method(self) -> None:
        with patch("datasets.translation_validator.TranslationValidator._build_llm"):
            self.v = TranslationValidator(threshold_accept=0.90, threshold_review=0.80)

    def test_accepted(self) -> None:
        assert self.v._decide(0.95) == "accepted"
        assert self.v._decide(0.90) == "accepted"

    def test_review(self) -> None:
        assert self.v._decide(0.89) == "review"
        assert self.v._decide(0.80) == "review"

    def test_rejected(self) -> None:
        assert self.v._decide(0.79) == "rejected"
        assert self.v._decide(0.50) == "rejected"


# ---------------------------------------------------------------------------
# TranslationValidator.validate — end-to-end (mocked)
# ---------------------------------------------------------------------------


class TestValidate:
    def test_report_has_correct_n(self, tmp_path: Path) -> None:
        csv_path = _make_csv(tmp_path)
        v = _mock_validator()
        report = v.validate(csv_path)
        assert report.n_total == 3

    def test_decisions_match_thresholds(self, tmp_path: Path) -> None:
        csv_path = _make_csv(tmp_path)
        # f1 = [0.95, 0.88, 0.72]
        v = _mock_validator(f1_scores=[0.95, 0.88, 0.72])
        report = v.validate(csv_path)
        decisions = [r["decision"] for r in report.rows]
        assert decisions == ["accepted", "review", "rejected"]

    def test_back_translate_called_once(self, tmp_path: Path) -> None:
        csv_path = _make_csv(tmp_path)
        v = _mock_validator()
        v.validate(csv_path)
        v.back_translate.assert_called_once()

    def test_bertscore_called_once(self, tmp_path: Path) -> None:
        csv_path = _make_csv(tmp_path)
        v = _mock_validator()
        v.validate(csv_path)
        v.compute_bertscore.assert_called_once()

    def test_report_contains_original_texts(self, tmp_path: Path) -> None:
        csv_path = _make_csv(tmp_path)
        v = _mock_validator()
        report = v.validate(csv_path)
        assert report.rows[0]["text_en_original"] == "The system shall process transactions."
        assert report.rows[0]["text_pt_translated"] == "O sistema deve processar transações."

    def test_report_contains_bertscore_columns(self, tmp_path: Path) -> None:
        csv_path = _make_csv(tmp_path)
        v = _mock_validator()
        report = v.validate(csv_path)
        row = report.rows[0]
        assert "bertscore_f1" in row
        assert "bertscore_precision" in row
        assert "bertscore_recall" in row

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame([{"RequirementText": "x", "class": "F"}]).to_csv(
            bad_csv, index=False, encoding="utf-8"
        )
        v = _mock_validator()
        with pytest.raises(ValueError, match="missing columns"):
            v.validate(bad_csv)

    def test_dropna_skips_empty_rows(self, tmp_path: Path) -> None:
        rows = [
            {
                "ProjectID": "P1",
                "RequirementText": "The system shall do X.",
                "RequirementText_PT": "O sistema deve fazer X.",
                "class": "F",
            },
            {
                "ProjectID": "P1",
                "RequirementText": None,
                "RequirementText_PT": "Tradução sem original.",
                "class": "F",
            },
        ]
        csv_path = _make_csv(tmp_path, rows=rows)
        v = _mock_validator(
            back_texts=["The system shall do X."],
            f1_scores=[0.95],
        )
        report = v.validate(csv_path)
        assert report.n_total == 1

    def test_all_accepted_no_warning(self, tmp_path: Path) -> None:
        csv_path = _make_csv(tmp_path)
        v = _mock_validator(f1_scores=[0.95, 0.92, 0.91])
        report = v.validate(csv_path)
        assert report.n_rejected == 0
        assert report.n_total == report.n_accepted

    def test_custom_thresholds(self, tmp_path: Path) -> None:
        csv_path = _make_csv(tmp_path)
        with patch("datasets.translation_validator.TranslationValidator._build_llm"):
            v = TranslationValidator(threshold_accept=0.95, threshold_review=0.85)
        v.back_translate = MagicMock(return_value=["a", "b", "c"])
        v.compute_bertscore = MagicMock(
            return_value=([0.96, 0.86, 0.74], [0.94, 0.84, 0.72], [0.95, 0.85, 0.73])
        )
        report = v.validate(csv_path)
        assert report.rows[0]["decision"] == "accepted"
        assert report.rows[1]["decision"] == "review"
        assert report.rows[2]["decision"] == "rejected"
