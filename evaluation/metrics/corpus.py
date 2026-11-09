"""Corpus metrics computed from a confusion matrix, fast enough to be recomputed on every
resample of a bootstrap or permutation test.

Classes are the integers 0..k-1. These agree with scikit-learn (Macro-F1 over the classes that
appear in the reference or the prediction, Cohen's kappa, multiclass Matthews coefficient) and
exist only so a test that recomputes the metric ten thousand times stays cheap.
"""

from __future__ import annotations

import numpy as np


def confusion(reference: np.ndarray, predicted: np.ndarray, k: int) -> np.ndarray:
    """Rows are the reference class, columns the predicted one."""
    flat = reference.astype(int) * k + predicted.astype(int)
    return np.bincount(flat, minlength=k * k).reshape(k, k).astype(float)


def accuracy(reference: np.ndarray, predicted: np.ndarray, k: int) -> float:
    return float(np.mean(reference == predicted))


def f1_macro(reference: np.ndarray, predicted: np.ndarray, k: int) -> float:
    cm = confusion(reference, predicted, k)
    true_positive = np.diag(cm)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    present = denominator > 0
    f1 = np.where(present, 2 * true_positive / np.where(present, denominator, 1), 0.0)
    return float(f1[present].mean())


def cohen_kappa(
    reference: np.ndarray, predicted: np.ndarray, k: int, weights: str | None = None
) -> float:
    """Nominal (`weights=None`) or linearly weighted kappa; NaN when chance agreement is perfect."""
    cm = confusion(reference, predicted, k)
    total = cm.sum()
    index = np.arange(k)
    distance = np.abs(index[:, None] - index[None, :]) / max(k - 1, 1)
    penalty = (1.0 - np.eye(k)) if weights is None else distance
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / total
    chance = float((penalty * expected).sum())
    if chance == 0.0:
        return float("nan")
    return 1.0 - float((penalty * cm).sum()) / chance


def matthews(reference: np.ndarray, predicted: np.ndarray, k: int) -> float:
    cm = confusion(reference, predicted, k)
    total = cm.sum()
    correct = np.trace(cm)
    per_reference, per_predicted = cm.sum(axis=1), cm.sum(axis=0)
    numerator = correct * total - float(per_reference @ per_predicted)
    denominator = np.sqrt(
        (total**2 - float(per_predicted @ per_predicted))
        * (total**2 - float(per_reference @ per_reference))
    )
    return 0.0 if denominator == 0 else float(numerator / denominator)
