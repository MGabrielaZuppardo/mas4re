from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from evaluation.runner import ClassifierRunResult

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path("experiments/results")
_LOGS_DIR = Path("experiments/logs")


def save_results(result: ClassifierRunResult, tag: str = "") -> Path:
    """
    Salva o resultado de uma rodada em JSON.

    Args:
        result: Resultado do runner.
        tag:    Tag opcional para identificar o experimento.

    Returns:
        Caminho do arquivo salvo.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_slug = result.model.replace("/", "_").replace(":", "-")
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    suffix = f"_{tag}" if tag else ""
    filename = f"{model_slug}_{result.dataset}{suffix}_{timestamp}.json"
    path = _RESULTS_DIR / filename

    payload = {
        "model": result.model,
        "dataset": result.dataset,
        "timestamp": timestamp,
        "tag": tag,
        "n_total": result.n_total,
        "n_classified": result.n_classified,
        "classification_rate": result.classification_rate,
        "elapsed_seconds": result.elapsed_seconds,
        "metrics": result.metrics,
        "subcategory_metrics": result.subcategory_metrics,
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resultado salvo | path=%s", path)
    return path


def print_summary(result: ClassifierRunResult) -> None:
    """Imprime um resumo legível no terminal."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  MODELO   : {result.model}")
    print(f"  DATASET  : {result.dataset}")
    print(f"  AMOSTRAS : {result.n_classified}/{result.n_total} classificados")
    print(f"  TEMPO    : {result.elapsed_seconds:.2f}s")
    print(sep)

    m = result.metrics
    if m:
        print(f"  Accuracy      : {m.get('accuracy', 'N/A')}")
        print(f"  F1 Macro      : {m.get('f1_macro', 'N/A')}")
        print(f"  F1 Weighted   : {m.get('f1_weighted', 'N/A')}")
        print(f"  MCC           : {m.get('mcc', 'N/A')}")

    sub = result.subcategory_metrics
    if sub:
        print(f"\n  {'Categoria':<12} {'F1':>6} {'Precision':>10} {'Recall':>8} {'Support':>8}")
        print(f"  {'-'*50}")
        for cat, vals in sub.items():
            if cat == "macro_avg":
                continue
            print(
                f"  {cat:<12} "
                f"{vals.get('f1', 0):>6.4f} "
                f"{vals.get('precision', 0):>10.4f} "
                f"{vals.get('recall', 0):>8.4f} "
                f"{int(vals.get('support', 0)):>8}"
            )
        macro = sub.get("macro_avg", {})
        if macro:
            print(f"  {'-'*50}")
            print(
                f"  {'macro_avg':<12} "
                f"{macro.get('f1', 0):>6.4f} "
                f"{macro.get('precision', 0):>10.4f} "
                f"{macro.get('recall', 0):>8.4f} "
                f"{int(macro.get('support', 0)):>8}"
            )
    print(sep + "\n")


def compare_models(results: list[ClassifierRunResult]) -> None:
    """Imprime tabela comparativa entre modelos."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  COMPARAÇÃO ENTRE MODELOS — PROMISE NFR+")
    print(sep)
    print(
        f"  {'Modelo':<30} {'Accuracy':>9} {'F1 Macro':>9} "
        f"{'F1 Weighted':>12} {'MCC':>7} {'Tempo':>8}"
    )
    print(f"  {'-'*76}")

    for r in sorted(results, key=lambda x: x.metrics.get("f1_macro", 0), reverse=True):
        m = r.metrics
        print(
            f"  {r.model:<30} "
            f"{m.get('accuracy', 0):>9.4f} "
            f"{m.get('f1_macro', 0):>9.4f} "
            f"{m.get('f1_weighted', 0):>12.4f} "
            f"{m.get('mcc', 0):>7.4f} "
            f"{r.elapsed_seconds:>7.1f}s"
        )
    print(sep + "\n")
