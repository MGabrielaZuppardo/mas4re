"""Statistical analysis for SQ2 and SQ3 (S4).

Provides three complementary tests used to compare MAS pipeline vs.
baseline across the 12 grid conditions (3 models x 2 languages x 2
architectures):

- Wilcoxon signed-rank test  : paired non-parametric significance test.
- Cliff's delta              : effect-size magnitude (no distribution assumption).
- Bootstrap CI               : 95 % confidence interval on the mean difference.

All functions accept plain Python lists of floats so they work directly
with the per-run metric values collected by ExperimentRunner.

Reference for effect-size thresholds (Romano et al. 2006):
    |d| < 0.147  → negligible
    |d| < 0.330  → small
    |d| < 0.474  → medium
    else         → large
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    significant: bool  # p < alpha
    alpha: float = 0.05


@dataclass(frozen=True)
class CliffsDeltaResult:
    delta: float
    magnitude: str  # "negligible" | "small" | "medium" | "large"


@dataclass(frozen=True)
class BootstrapCIResult:
    mean_difference: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    confidence: float  # e.g. 0.95


@dataclass(frozen=True)
class PairwiseReport:
    """Full statistical comparison between two paired samples."""

    metric_name: str
    n: int
    mean_a: float
    mean_b: float
    wilcoxon: WilcoxonResult
    cliffs_delta: CliffsDeltaResult
    bootstrap_ci: BootstrapCIResult


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------


def wilcoxon_test(
    a: list[float],
    b: list[float],
    alpha: float = 0.05,
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test between samples a and b.

    Null hypothesis: the distribution of differences (a - b) is symmetric
    about zero. Ties are handled with the 'wilcox' zero-method (dropped).

    Args:
        a: Metric values from strategy A (e.g. pipeline) — one per run.
        b: Metric values from strategy B (e.g. baseline) — one per run.
        alpha: Significance threshold (default 0.05).

    Raises:
        ValueError: if len(a) != len(b) or n < 2.
    """
    if len(a) != len(b):
        raise ValueError(f"Samples must be paired (len a={len(a)}, len b={len(b)})")
    if len(a) < 2:
        raise ValueError("Need at least 2 paired observations")

    diff = np.array(a) - np.array(b)
    non_zero = diff[diff != 0]
    if len(non_zero) == 0:
        return WilcoxonResult(statistic=0.0, p_value=1.0, significant=False, alpha=alpha)

    stat, p = wilcoxon(non_zero, zero_method="wilcox", alternative="two-sided")
    return WilcoxonResult(
        statistic=float(stat),
        p_value=float(p),
        significant=bool(p < alpha),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Cliff's delta
# ---------------------------------------------------------------------------


def _magnitude(delta: float) -> str:
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "negligible"
    if abs_d < 0.330:
        return "small"
    if abs_d < 0.474:
        return "medium"
    return "large"


def cliffs_delta(a: list[float], b: list[float]) -> CliffsDeltaResult:
    """Non-parametric effect size: proportion of (a > b) minus (a < b) pairs.

    d =  1.0  → a always dominates b
    d = -1.0  → b always dominates a
    d =  0.0  → no systematic difference

    Args:
        a: Metric values from strategy A.
        b: Metric values from strategy B (need not be paired or same length).
    """
    if not a or not b:
        raise ValueError("Both samples must be non-empty")

    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)

    greater = float(np.sum(arr_a[:, None] > arr_b[None, :]))
    less = float(np.sum(arr_a[:, None] < arr_b[None, :]))
    n_pairs = len(arr_a) * len(arr_b)

    delta = (greater - less) / n_pairs
    return CliffsDeltaResult(delta=round(delta, 4), magnitude=_magnitude(delta))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval on mean difference
# ---------------------------------------------------------------------------


def bootstrap_ci(
    a: list[float],
    b: list[float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCIResult:
    """Bootstrap 95 % CI on the mean difference (a - b).

    Uses the percentile method: resample pairs with replacement, compute
    mean(a*) - mean(b*) for each replicate, take the (α/2, 1-α/2) quantiles.

    Args:
        a: Metric values from strategy A (paired with b).
        b: Metric values from strategy B (paired with a).
        n_bootstrap: Number of bootstrap replicates (default 10 000).
        confidence: Desired confidence level (default 0.95).
        seed: Random seed for reproducibility.
    """
    if len(a) != len(b):
        raise ValueError(f"Samples must be paired (len a={len(a)}, len b={len(b)})")
    if len(a) < 2:
        raise ValueError("Need at least 2 paired observations")

    rng = np.random.default_rng(seed)
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    n = len(arr_a)

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = arr_a[idx].mean() - arr_b[idx].mean()

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    return BootstrapCIResult(
        mean_difference=round(float(arr_a.mean() - arr_b.mean()), 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Full pairwise report (convenience wrapper for notebooks)
# ---------------------------------------------------------------------------


def pairwise_report(
    metric_name: str,
    a: list[float],
    b: list[float],
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> PairwiseReport:
    """Run all three tests and return a single report object.

    Intended for use in notebooks and LaTeX table export::

        report = pairwise_report("f1_macro", pipeline_f1s, baseline_f1s)
        print(report.wilcoxon.p_value)
        print(report.cliffs_delta.magnitude)

    Args:
        metric_name: Human-readable label (e.g. "f1_macro", "accuracy").
        a: Values from strategy A (pipeline / MAS).
        b: Values from strategy B (baseline).
        alpha: Significance threshold for Wilcoxon.
        n_bootstrap: Bootstrap replicates.
        seed: Random seed.
    """
    return PairwiseReport(
        metric_name=metric_name,
        n=len(a),
        mean_a=round(float(np.mean(a)), 6),
        mean_b=round(float(np.mean(b)), 6),
        wilcoxon=wilcoxon_test(a, b, alpha=alpha),
        cliffs_delta=cliffs_delta(a, b),
        bootstrap_ci=bootstrap_ci(a, b, n_bootstrap=n_bootstrap, seed=seed),
    )
