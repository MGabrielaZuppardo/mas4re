from __future__ import annotations

import logging

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)

from domain.enums import NFRCategory, RequirementType
from domain.models import ClassifiedRequirement, Requirement

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    predictions: list[ClassifiedRequirement],
    ground_truth: list[Requirement],
) -> dict[str, float]:
    """
    Calcula métricas de classificação FR/NFR.

    Métricas:
        - accuracy:   Acurácia geral (F vs NF).
        - f1_macro:   F1 macro (trata classes igualmente).
        - f1_weighted: F1 ponderado pelo suporte de cada classe.
        - mcc:        Matthews Correlation Coefficient.

    Args:
        predictions:  Requisitos classificados pelo agente.
        ground_truth: Requisitos com label real em metadata["label_type"].

    Returns:
        Dicionário com as métricas calculadas.
    """
    gt_map = {r.id: r for r in ground_truth}
    common_ids = [r.id for r in predictions if r.id in gt_map]

    if not common_ids:
        logger.warning("Nenhum ID em comum para compute_classification_metrics.")
        return {}

    y_true = [gt_map[i].metadata["label_type"] for i in common_ids]
    y_pred = [next(r.requirement_type.value for r in predictions if r.id == i) for i in common_ids]

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "n_evaluated": len(common_ids),
    }
    logger.info("Métricas de classificação: %s", metrics)
    return metrics


def compute_subcategory_metrics(
    predictions: list[ClassifiedRequirement],
    ground_truth: list[Requirement],
) -> dict[str, dict[str, float]]:
    """
    Calcula métricas por subcategoria NFR.

    Usa average='macro' para tratar cada categoria igualmente,
    independentemente do suporte.

    Returns:
        Dicionário {categoria: {f1, precision, recall, support}}
    """
    gt_map = {r.id: r for r in ground_truth}

    # Filtra apenas NFRs com label de categoria
    nfr_preds = [
        r
        for r in predictions
        if r.id in gt_map
        and gt_map[r.id].metadata.get("label_category")
        and r.requirement_type == RequirementType.NON_FUNCTIONAL
    ]

    if not nfr_preds:
        logger.warning("Nenhum NFR encontrado para compute_subcategory_metrics.")
        return {}

    y_true = [gt_map[r.id].metadata["label_category"] for r in nfr_preds]
    y_pred = [r.nfr_category if r.nfr_category else "F" for r in nfr_preds]

    report = classification_report(
        y_true,
        y_pred,
        labels=[c.value for c in NFRCategory.nfr_only()],
        output_dict=True,
        zero_division=0,
    )

    result: dict[str, dict[str, float]] = {}
    for cat in NFRCategory.nfr_only():
        key = cat.value
        if key in report:
            result[key] = {
                "f1": round(report[key]["f1-score"], 4),
                "precision": round(report[key]["precision"], 4),
                "recall": round(report[key]["recall"], 4),
                "support": int(report[key]["support"]),
            }

    # Macro médio geral das subcategorias
    macro = report.get("macro avg", {})
    result["macro_avg"] = {
        "f1": round(macro.get("f1-score", 0.0), 4),
        "precision": round(macro.get("precision", 0.0), 4),
        "recall": round(macro.get("recall", 0.0), 4),
        "support": int(macro.get("support", 0)),
    }

    return result
