from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from scipy.stats import kendalltau, spearmanr

from domain.enums import MoSCoWPriority
from domain.models import PrioritizedRequirement

logger = logging.getLogger(__name__)


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
    gt_ranks   = np.array([gt_map[i].priority_rank or 0 for i in common_ids], dtype=float)
    pred_scores = np.array([pred_map[i].priority_score or 0.0 for i in common_ids])
    gt_scores   = np.array([gt_map[i].priority_score or 0.0 for i in common_ids])

    tau, _ = kendalltau(pred_ranks, gt_ranks)
    rho, _ = spearmanr(pred_scores, gt_scores)
    mae    = float(np.mean(np.abs(pred_scores - gt_scores)))

    moscow_acc = sum(
        pred_map[i].priority == gt_map[i].priority
        for i in common_ids
    ) / len(common_ids)

    metrics = {
        "kendall_tau":      round(float(tau), 4),
        "spearman_r":       round(float(rho), 4),
        "mae_score":        round(mae, 4),
        "moscow_accuracy":  round(moscow_acc, 4),
        "n_evaluated":      len(common_ids),
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

    counts = Counter(
        r.priority.value for r in requirements if r.priority is not None
    )
    return {
        cat.value: round(counts.get(cat.value, 0) / total, 4)
        for cat in MoSCoWPriority
    }