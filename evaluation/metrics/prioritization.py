from __future__ import annotations

import logging
import math
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from domain.enums import MoSCoWPriority
from domain.models import PrioritizedRequirement, Requirement

logger = logging.getLogger(__name__)

_ORDER = (
    MoSCoWPriority.WONT_HAVE,
    MoSCoWPriority.COULD_HAVE,
    MoSCoWPriority.SHOULD_HAVE,
    MoSCoWPriority.MUST_HAVE,
)
_ORDINALS = [priority.ordinal for priority in _ORDER]


def _finite(value: float) -> float | None:
    """JSON has no NaN: an undefined statistic (constant input) is reported as null."""
    number = float(value)
    return None if math.isnan(number) else round(number, 4)


def _quietly(statistic: Callable[[], float]) -> float | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _finite(statistic())


def _score_of(item: PrioritizedRequirement, priority: MoSCoWPriority) -> float:
    return item.priority_score if item.priority_score is not None else priority.score


def _correlation(
    method: Callable[..., Any], x: Sequence[float], y: Sequence[float]
) -> float | None:
    return _quietly(lambda: method(x, y)[0])


def _per_class(y_true: list[int], y_pred: list[int]) -> dict[str, dict[str, float | int]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=_ORDINALS, zero_division=0
    )
    predicted = Counter(y_pred)
    return {
        priority.value: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
            "predicted": predicted[priority.ordinal],
        }
        for i, priority in enumerate(_ORDER)
    }


def _confusion(y_true: list[int], y_pred: list[int]) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(y_true, y_pred, labels=_ORDINALS)
    return {
        gold.value: {pred.value: int(matrix[i][j]) for j, pred in enumerate(_ORDER)}
        for i, gold in enumerate(_ORDER)
    }


def compute_priority_gold_metrics(
    prioritized: Sequence[PrioritizedRequirement],
    requirements: Sequence[Requirement],
) -> dict[str, Any]:
    """Priorities against the human MoSCoW labels of the dataset.

    The gold lives in `Requirement.metadata["ground_truth_priority"]`; datasets without it get
    an empty result. Requirements the agents could not process are not in `prioritized`, so
    `coverage` is reported next to every figure. MoSCoW is ordinal, hence the weighted kappa and
    the rank correlations next to accuracy; `majority_baseline_accuracy` is what always answering
    the most frequent class would score on the same items.
    """
    gold = {r.id: g for r in requirements if (g := r.gold_priority) is not None}
    if not gold:
        return {}
    relevance = {r.id: r.metadata.get("ground_truth_relevance") for r in requirements}
    pairs = [
        (p.id, gold[p.id], p.priority, _score_of(p, p.priority))
        for p in prioritized
        if p.id in gold and p.priority is not None
    ]
    result: dict[str, Any] = {
        "n_total": len(gold),
        "n_evaluated": len(pairs),
        "coverage": round(len(pairs) / len(gold), 4),
    }
    if not pairs:
        return result

    y_true = [g.ordinal for _, g, _, _ in pairs]
    y_pred = [predicted.ordinal for _, _, predicted, _ in pairs]
    scores = [score for _, _, _, score in pairs]
    result.update(
        accuracy=round(float(np.mean(np.array(y_true) == np.array(y_pred))), 4),
        majority_baseline_accuracy=round(max(Counter(y_true).values()) / len(y_true), 4),
        weighted_kappa=_quietly(
            lambda: cohen_kappa_score(y_true, y_pred, weights="linear", labels=_ORDINALS)
        ),
        mae_ordinal=round(float(np.mean(np.abs(np.array(y_true) - np.array(y_pred)))), 4),
        spearman=_correlation(spearmanr, y_true, y_pred),
        kendall_tau_b=_correlation(kendalltau, y_true, y_pred),
        spearman_score=_correlation(spearmanr, y_true, scores),
        kendall_tau_b_score=_correlation(kendalltau, y_true, scores),
        per_class=_per_class(y_true, y_pred),
        confusion_matrix=_confusion(y_true, y_pred),
    )
    rated = [
        (float(rating), score)
        for item_id, _, _, score in pairs
        if (rating := relevance[item_id]) is not None
    ]
    if rated:
        result["spearman_score_vs_relevance"] = _correlation(
            spearmanr, [r for r, _ in rated], [s for _, s in rated]
        )
    return result


def compute_prioritization_metrics(
    predictions: list[PrioritizedRequirement],
    ground_truth: list[PrioritizedRequirement],
) -> dict[str, float]:
    """
    Calcula métricas de qualidade de priorização.

    Métricas:
        - kendall_tau:   Correlação de Kendall entre rankings previstos e reais.
        - spearman_r:    Correlação de Spearman entre scores previstos e reais.
        - mae_score:     Erro absoluto médio entre priority_scores.
        - moscow_accuracy: Acurácia exata de categoria MoSCoW.

    Args:
        predictions:  Lista de requisitos priorizados pelo agente.
        ground_truth: Lista de requisitos priorizados de referência (ground truth).

    Returns:
        Dicionário com as métricas calculadas.
    """
    if not predictions or not ground_truth:
        logger.warning("Listas vazias fornecidas para compute_prioritization_metrics.")
        return {}

    # Alinha pelo ID
    gt_map = {r.id: r for r in ground_truth}
    pred_map = {r.id: r for r in predictions}
    common_ids = [r.id for r in predictions if r.id in gt_map]

    if len(common_ids) < 2:
        logger.warning("IDs insuficientes para calcular correlações (n=%d).", len(common_ids))
        return {}

    pred_ranks = np.array([pred_map[i].priority_rank or 0 for i in common_ids], dtype=float)
    gt_ranks = np.array([gt_map[i].priority_rank or 0 for i in common_ids], dtype=float)
    pred_scores = np.array([pred_map[i].priority_score or 0.0 for i in common_ids])
    gt_scores = np.array([gt_map[i].priority_score or 0.0 for i in common_ids])

    tau, _ = kendalltau(pred_ranks, gt_ranks)
    rho, _ = spearmanr(pred_scores, gt_scores)
    mae = float(np.mean(np.abs(pred_scores - gt_scores)))

    moscow_acc = sum(pred_map[i].priority == gt_map[i].priority for i in common_ids) / len(
        common_ids
    )

    metrics = {
        "kendall_tau": round(float(tau), 4),
        "spearman_r": round(float(rho), 4),
        "mae_score": round(mae, 4),
        "moscow_accuracy": round(moscow_acc, 4),
        "n_evaluated": len(common_ids),
    }
    logger.info("Métricas de priorização: %s", metrics)
    return metrics


def compute_moscow_distribution(
    requirements: list[PrioritizedRequirement],
) -> dict[str, float]:
    """
    Calcula a distribuição percentual das categorias MoSCoW.

    Returns:
        Dicionário {categoria: proporção} — ex: {"M": 0.30, "S": 0.40, ...}
    """
    total = len(requirements)
    if total == 0:
        return {}

    counts = Counter(r.priority.value for r in requirements if r.priority is not None)
    return {cat.value: round(counts.get(cat.value, 0) / total, 4) for cat in MoSCoWPriority}
