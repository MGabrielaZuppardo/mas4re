from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from agents.classifier import ClassificationAgent
from datasets.promise import PromiseAdapter
from domain.models import ClassifiedRequirement, Requirement
from evaluation.metrics.classification import (
    compute_classification_metrics,
    compute_subcategory_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassifierRunResult:
    """Resultado de uma rodada do classificador num dataset."""

    model: str
    dataset: str
    n_total: int
    n_classified: int
    elapsed_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)
    subcategory_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    predictions: list[ClassifiedRequirement] = field(default_factory=list)

    @property
    def classification_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return round(self.n_classified / self.n_total, 4)


def run_classifier_on_promise(
    model: str,
    n_samples: int | None = None,
    seed: int = 42,
    max_workers: int = 3,
    temperature: float = 0.0,
) -> ClassifierRunResult:
    """
    Roda o ClassificationAgent no dataset PROMISE NFR+.

    Args:
        model:       Identificador do modelo (ex: "ollama/qwen2.5:7b").
        n_samples:   Número de requisitos a amostrar. None = dataset completo.
        seed:        Seed para reprodutibilidade da amostra.
        max_workers: Paralelismo do classify_batch.
        temperature: Temperatura do LLM.

    Returns:
        ClassifierRunResult com métricas calculadas.
    """
    logger.info(
        "Iniciando avaliação | model=%s | n_samples=%s | seed=%d",
        model,
        n_samples or "all",
        seed,
    )

    # Carrega dataset
    adapter = PromiseAdapter()
    requirements: list[Requirement] = (
        adapter.load_sample(n_samples, seed=seed) if n_samples else adapter.load()
    )

    logger.info("Dataset carregado | n=%d", len(requirements))

    # Roda classificador
    agent = ClassificationAgent(model=model, temperature=temperature)

    start = time.perf_counter()
    predictions = agent.classify_batch(requirements, max_workers=max_workers)
    elapsed = round(time.perf_counter() - start, 2)

    logger.info(
        "Classificação concluída | n_classificados=%d | elapsed=%.2fs",
        len(predictions),
        elapsed,
    )

    # Calcula métricas
    metrics = compute_classification_metrics(predictions, requirements)
    subcategory_metrics = compute_subcategory_metrics(predictions, requirements)

    return ClassifierRunResult(
        model=model,
        dataset="PROMISE_NFR_PT",
        n_total=len(requirements),
        n_classified=len(predictions),
        elapsed_seconds=elapsed,
        metrics=metrics,
        subcategory_metrics=subcategory_metrics,
        predictions=predictions,
    )
