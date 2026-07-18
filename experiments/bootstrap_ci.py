"""Shared paired bootstrap CI + p-value math for MAS4RE's RQ2/RQ3 scripts.

Extracted from experiments/compute_ablation_bootstrap.py (RQ2, ΔF1-macro) and
experiments/compute_rq3_kappa_bootstrap.py (RQ3, Δkappa) — both implemented
the exact same percentile-bootstrap CI + two-sided bootstrap p-value formula
on an already-resampled array of paired deltas. Kept here so the two RQs
can't silently diverge if the formula is ever adjusted.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    observed_delta: float,
    deltas: np.ndarray,
    alpha: float,
    n_comparisons: int,
) -> dict:
    """95% and Bonferroni-adjusted percentile bootstrap CI, plus a two-sided
    bootstrap p-value, for an observed paired delta given its bootstrap
    resample distribution.

    Args:
        observed_delta: the delta computed on the real (non-resampled) data.
        deltas: array of per-resample deltas (same length as n_boot).
        alpha: family-wise alpha before Bonferroni correction (e.g. 0.05).
        n_comparisons: number of comparisons sharing this alpha budget.
    """
    alpha_bonf = alpha / n_comparisons
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    bonf_pct = [100 * alpha_bonf / 2, 100 * (1 - alpha_bonf / 2)]
    ci_bonf_low, ci_bonf_high = np.percentile(deltas, bonf_pct)
    # Fraction of resamples on the opposite side of 0 from the observed
    # delta (two-sided) — a bootstrap-based analogue to a p-value.
    if observed_delta >= 0:
        p_boot = float(np.mean(deltas <= 0)) * 2
    else:
        p_boot = float(np.mean(deltas >= 0)) * 2
    p_boot = min(p_boot, 1.0)

    return {
        "ci95_low": round(float(ci_low), 4),
        "ci95_high": round(float(ci_high), 4),
        "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
        "ci_bonferroni_low": round(float(ci_bonf_low), 4),
        "ci_bonferroni_high": round(float(ci_bonf_high), 4),
        "ci_bonferroni_includes_zero": bool(ci_bonf_low <= 0 <= ci_bonf_high),
        "p_bootstrap_two_sided": round(p_boot, 4),
    }
