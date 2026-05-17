"""Unit tests for evaluation/metrics/statistical.py (PR #23).

All tests are deterministic (seed=42) and run without LLM or Ollama.
Covers: WilcoxonResult, CliffsDeltaResult, BootstrapCIResult, PairwiseReport,
plus edge cases (ties, identical samples, minimum n, wrong lengths).
"""

from __future__ import annotations

import math

import pytest

from evaluation.metrics.statistical import (
    BootstrapCIResult,
    CliffsDeltaResult,
    PairwiseReport,
    WilcoxonResult,
    bootstrap_ci,
    cliffs_delta,
    pairwise_report,
    wilcoxon_test,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic metric lists that mimic real grid runs (n=10 runs)
# ---------------------------------------------------------------------------

PIPELINE_F1 = [0.91, 0.89, 0.90, 0.88, 0.92, 0.87, 0.93, 0.90, 0.89, 0.91]
BASELINE_F1 = [0.82, 0.80, 0.83, 0.79, 0.84, 0.78, 0.85, 0.81, 0.80, 0.82]
IDENTICAL = [0.85] * 10
TINY = [0.90, 0.80]  # minimum n=2


# ---------------------------------------------------------------------------
# wilcoxon_test
# ---------------------------------------------------------------------------


class TestWilcoxonTest:
    def test_returns_wilcoxon_result(self) -> None:
        result = wilcoxon_test(PIPELINE_F1, BASELINE_F1)
        assert isinstance(result, WilcoxonResult)

    def test_significant_when_pipeline_clearly_better(self) -> None:
        result = wilcoxon_test(PIPELINE_F1, BASELINE_F1)
        assert result.significant
        assert result.p_value < 0.05

    def test_not_significant_for_identical_samples(self) -> None:
        result = wilcoxon_test(IDENTICAL, IDENTICAL)
        assert not result.significant
        assert result.p_value == 1.0

    def test_statistic_is_non_negative(self) -> None:
        result = wilcoxon_test(PIPELINE_F1, BASELINE_F1)
        assert result.statistic >= 0

    def test_p_value_in_unit_interval(self) -> None:
        result = wilcoxon_test(PIPELINE_F1, BASELINE_F1)
        assert 0.0 <= result.p_value <= 1.0

    def test_alpha_stored_in_result(self) -> None:
        result = wilcoxon_test(PIPELINE_F1, BASELINE_F1, alpha=0.01)
        assert result.alpha == 0.01

    def test_significance_respects_alpha(self) -> None:
        """With alpha=1.0 every result should be significant."""
        result = wilcoxon_test(PIPELINE_F1, BASELINE_F1, alpha=1.0)
        assert result.significant

    def test_minimum_n_two(self) -> None:
        result = wilcoxon_test(TINY, [0.70, 0.60])
        assert isinstance(result, WilcoxonResult)

    def test_raises_on_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="paired"):
            wilcoxon_test([0.9, 0.8], [0.7])

    def test_raises_on_single_observation(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            wilcoxon_test([0.9], [0.8])

    def test_symmetric_difference_gives_p_one(self) -> None:
        """All differences zero → p_value == 1.0."""
        result = wilcoxon_test([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
        assert result.p_value == 1.0


# ---------------------------------------------------------------------------
# cliffs_delta
# ---------------------------------------------------------------------------


class TestCliffsDelta:
    def test_returns_cliffs_delta_result(self) -> None:
        result = cliffs_delta(PIPELINE_F1, BASELINE_F1)
        assert isinstance(result, CliffsDeltaResult)

    def test_delta_in_minus_one_to_one(self) -> None:
        result = cliffs_delta(PIPELINE_F1, BASELINE_F1)
        assert -1.0 <= result.delta <= 1.0

    def test_large_effect_when_a_dominates(self) -> None:
        a = [1.0] * 10
        b = [0.0] * 10
        result = cliffs_delta(a, b)
        assert result.delta == pytest.approx(1.0)
        assert result.magnitude == "large"

    def test_large_effect_when_b_dominates(self) -> None:
        a = [0.0] * 10
        b = [1.0] * 10
        result = cliffs_delta(a, b)
        assert result.delta == pytest.approx(-1.0)
        assert result.magnitude == "large"

    def test_negligible_for_identical(self) -> None:
        result = cliffs_delta(IDENTICAL, IDENTICAL)
        assert result.delta == pytest.approx(0.0)
        assert result.magnitude == "negligible"

    def test_magnitude_small(self) -> None:
        # 6 high + 4 low vs 10 mid → delta = (60-40)/100 = 0.20 (small)
        a = [0.60] * 6 + [0.40] * 4
        b = [0.50] * 10
        result = cliffs_delta(a, b)
        assert result.magnitude == "small"

    def test_magnitude_medium(self) -> None:
        # 7 high + 3 low vs 10 mid → delta = (70-30)/100 = 0.40 (medium)
        a = [0.60] * 7 + [0.40] * 3
        b = [0.50] * 10
        result = cliffs_delta(a, b)
        assert result.magnitude == "medium"

    def test_raises_on_empty_a(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            cliffs_delta([], BASELINE_F1)

    def test_raises_on_empty_b(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            cliffs_delta(PIPELINE_F1, [])

    def test_unequal_lengths_allowed(self) -> None:
        """Cliff's delta does not require paired samples."""
        result = cliffs_delta([0.9, 0.8, 0.7], [0.6, 0.5])
        assert isinstance(result, CliffsDeltaResult)

    def test_pipeline_vs_baseline_is_large(self) -> None:
        result = cliffs_delta(PIPELINE_F1, BASELINE_F1)
        assert result.magnitude == "large"


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_bootstrap_ci_result(self) -> None:
        result = bootstrap_ci(PIPELINE_F1, BASELINE_F1, seed=42)
        assert isinstance(result, BootstrapCIResult)

    def test_ci_lower_le_upper(self) -> None:
        result = bootstrap_ci(PIPELINE_F1, BASELINE_F1, seed=42)
        assert result.ci_lower <= result.ci_upper

    def test_mean_difference_within_ci(self) -> None:
        result = bootstrap_ci(PIPELINE_F1, BASELINE_F1, seed=42)
        assert result.ci_lower <= result.mean_difference <= result.ci_upper

    def test_positive_mean_difference_when_a_better(self) -> None:
        result = bootstrap_ci(PIPELINE_F1, BASELINE_F1, seed=42)
        assert result.mean_difference > 0

    def test_ci_excludes_zero_when_clearly_different(self) -> None:
        result = bootstrap_ci(PIPELINE_F1, BASELINE_F1, seed=42)
        assert result.ci_lower > 0, "CI should exclude 0 when pipeline clearly beats baseline"

    def test_ci_includes_zero_for_identical(self) -> None:
        result = bootstrap_ci(IDENTICAL, IDENTICAL, seed=42)
        assert result.ci_lower <= 0 <= result.ci_upper

    def test_n_bootstrap_stored(self) -> None:
        result = bootstrap_ci(PIPELINE_F1, BASELINE_F1, n_bootstrap=500, seed=42)
        assert result.n_bootstrap == 500

    def test_confidence_stored(self) -> None:
        result = bootstrap_ci(PIPELINE_F1, BASELINE_F1, confidence=0.99, seed=42)
        assert result.confidence == 0.99

    def test_reproducible_with_same_seed(self) -> None:
        r1 = bootstrap_ci(PIPELINE_F1, BASELINE_F1, seed=42)
        r2 = bootstrap_ci(PIPELINE_F1, BASELINE_F1, seed=42)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_raises_on_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="paired"):
            bootstrap_ci([0.9, 0.8], [0.7])

    def test_raises_on_single_observation(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_ci([0.9], [0.8])


# ---------------------------------------------------------------------------
# pairwise_report
# ---------------------------------------------------------------------------


class TestPairwiseReport:
    def test_returns_pairwise_report(self) -> None:
        result = pairwise_report("f1_macro", PIPELINE_F1, BASELINE_F1)
        assert isinstance(result, PairwiseReport)

    def test_metric_name_stored(self) -> None:
        result = pairwise_report("accuracy", PIPELINE_F1, BASELINE_F1)
        assert result.metric_name == "accuracy"

    def test_n_equals_len_a(self) -> None:
        result = pairwise_report("f1_macro", PIPELINE_F1, BASELINE_F1)
        assert result.n == len(PIPELINE_F1)

    def test_mean_a_and_b_correct(self) -> None:
        result = pairwise_report("f1_macro", PIPELINE_F1, BASELINE_F1)
        assert result.mean_a == pytest.approx(sum(PIPELINE_F1) / len(PIPELINE_F1), rel=1e-4)
        assert result.mean_b == pytest.approx(sum(BASELINE_F1) / len(BASELINE_F1), rel=1e-4)

    def test_mean_a_greater_than_mean_b(self) -> None:
        result = pairwise_report("f1_macro", PIPELINE_F1, BASELINE_F1)
        assert result.mean_a > result.mean_b

    def test_sub_results_are_correct_types(self) -> None:
        result = pairwise_report("f1_macro", PIPELINE_F1, BASELINE_F1)
        assert isinstance(result.wilcoxon, WilcoxonResult)
        assert isinstance(result.cliffs_delta, CliffsDeltaResult)
        assert isinstance(result.bootstrap_ci, BootstrapCIResult)

    def test_full_report_pipeline_vs_baseline(self) -> None:
        result = pairwise_report("f1_macro", PIPELINE_F1, BASELINE_F1, seed=42)
        assert result.wilcoxon.significant
        assert result.cliffs_delta.magnitude == "large"
        assert result.bootstrap_ci.ci_lower > 0

    def test_no_nan_in_report(self) -> None:
        result = pairwise_report("f1_macro", PIPELINE_F1, BASELINE_F1, seed=42)
        assert not math.isnan(result.mean_a)
        assert not math.isnan(result.mean_b)
        assert not math.isnan(result.wilcoxon.p_value)
        assert not math.isnan(result.cliffs_delta.delta)
        assert not math.isnan(result.bootstrap_ci.mean_difference)
