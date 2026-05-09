"""
Script de execução do pipeline MAS (ClassificationAgent + PrioritizationAgent)
sobre o dataset PROMISE NFR+.

Uso básico (amostra de 10 requisitos, modelo Groq):
    python scripts/run_mas_pipeline.py

Exemplos:
    python scripts/run_mas_pipeline.py --n 20
    python scripts/run_mas_pipeline.py --full --clf-model claude-haiku-4-5
    python scripts/run_mas_pipeline.py --n 10 --clf-model ollama/qwen2.5:7b --pri-model ollama/llama3.1:8b
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_mas_pipeline")

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from config.settings import settings
from datasets.promise import PromiseAdapter
from domain.models import PipelineState
from evaluation.metrics.classification import (
    compute_classification_metrics,
    compute_subcategory_metrics,
)
from evaluation.metrics.prioritization import compute_moscow_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline MAS (classificador + priorizador) no PROMISE NFR+"
    )
    parser.add_argument(
        "--clf-model",
        default=settings.model_groq_default,
        help="Modelo do ClassificationAgent",
    )
    parser.add_argument(
        "--pri-model",
        default=settings.model_groq_default,
        help="Modelo do PrioritizationAgent (padrão: mesmo do classificador)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Número de requisitos da amostra (ignorado se --full)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade da amostra",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Roda no dataset completo (~625 requisitos)",
    )
    parser.add_argument(
        "--out",
        default="experiments/results",
        help="Diretório de saída dos resultados",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Dataset ───────────────────────────────────────────────────────────────
    adapter = PromiseAdapter(path=settings.promise_dataset_path)

    if args.full:
        logger.info("Carregando dataset completo...")
        requirements = adapter.load()
    else:
        logger.info("Carregando amostra | n=%d | seed=%d", args.n, args.seed)
        requirements = adapter.load_sample(n=args.n, seed=args.seed)

    logger.info("Requisitos carregados: %d", len(requirements))

    # ── Agentes ───────────────────────────────────────────────────────────────
    classifier  = ClassificationAgent(model=args.clf_model)
    prioritizer = PrioritizationAgent(model=args.pri_model)

    logger.info("Agentes criados | clf=%s | pri=%s", args.clf_model, args.pri_model)

    # ── Execução ──────────────────────────────────────────────────────────────
    state = PipelineState(raw_requirements=requirements)

    logger.info("── Etapa 1: Classificação ──")
    started_at = datetime.now()
    state = classifier.run(state)
    clf_elapsed = (datetime.now() - started_at).total_seconds()
    logger.info("Classificação concluída em %.1fs", clf_elapsed)

    logger.info("── Etapa 2: Priorização ──")
    pri_start = datetime.now()
    state = prioritizer.run(state)
    pri_elapsed = (datetime.now() - pri_start).total_seconds()
    total_elapsed = (datetime.now() - started_at).total_seconds()
    logger.info("Priorização concluída em %.1fs", pri_elapsed)

    classified  = state.classified_requirements
    prioritized = state.prioritized_requirements

    # ── Métricas ──────────────────────────────────────────────────────────────
    # Classificação — ground truth disponível no PROMISE
    clf_metrics    = compute_classification_metrics(classified, requirements)
    subcat_metrics = compute_subcategory_metrics(classified, requirements)

    # Priorização — sem ground truth no PROMISE, só distribuição
    moscow_dist = compute_moscow_distribution(prioritized)

    # ── Exibe no console ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  PIPELINE MAS | clf={args.clf_model} | pri={args.pri_model}")
    print(f"  n={len(prioritized)} requisitos | {total_elapsed:.1f}s total")
    print("=" * 70)

    print("\n── Predições ──────────────────────────────────────────────────────")
    for req in prioritized:
        gt_type = req.metadata.get("label_type", "?")
        gt_cat  = req.metadata.get("label_category", "?")
        pred_type = req.requirement_type.value if req.requirement_type else "?"
        pred_cat  = req.nfr_category.value if req.nfr_category else "-"

        match_type = "✓" if gt_type == pred_type else "✗"
        match_cat  = "✓" if gt_cat == pred_cat else "✗"

        print(
            f"[{req.priority_rank:>3}] {req.priority} {req.priority_score:.2f} | "
            f"tipo {match_type} GT={gt_type} Pred={pred_type} | "
            f"cat {match_cat} GT={gt_cat} Pred={pred_cat} | "
            f"{req.text[:50]}..."
        )

    print("\n── Métricas de Classificação ──────────────────────────────────────")
    for k, v in clf_metrics.items():
        print(f"  {k:<20} {v}")

    print("\n── Distribuição MoSCoW ────────────────────────────────────────────")
    for cat, pct in moscow_dist.items():
        bar = "█" * int(pct * 30)
        print(f"  {cat}  {bar:<30} {pct:.1%}")

    print("\n── Métricas por Subcategoria NFR ──────────────────────────────────")
    for cat, m in subcat_metrics.items():
        if cat == "macro_avg":
            continue
        print(f"  {cat:<4} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  n={m['support']}")
    macro = subcat_metrics.get("macro_avg", {})
    if macro:
        print(f"\n  macro_avg F1={macro['f1']:.3f}  P={macro['precision']:.3f}  R={macro['recall']:.3f}")

    print(f"\n  clf_elapsed={clf_elapsed:.1f}s | pri_elapsed={pri_elapsed:.1f}s | total={total_elapsed:.1f}s")
    print("=" * 70)

    # ── Salva JSON ────────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode      = "full" if args.full else f"sample{args.n}"
    clf_tag   = args.clf_model.replace("/", "-")
    out_file  = out_dir / f"mas_pipeline_{clf_tag}_{mode}_{timestamp}.json"

    payload = {
        "run_id":          state.run_id,
        "clf_model":       args.clf_model,
        "pri_model":       args.pri_model,
        "dataset":         "PROMISE_NFR_PT",
        "n":               len(prioritized),
        "elapsed": {
            "classification_seconds":  round(clf_elapsed, 2),
            "prioritization_seconds":  round(pri_elapsed, 2),
            "total_seconds":           round(total_elapsed, 2),
        },
        # ── Métricas ──────────────────────────────────────────────────────────
        "metrics": {
            "classification":      clf_metrics,
            "subcategory":         subcat_metrics,
            "moscow_distribution": moscow_dist,
            # Nota: métricas de correlação de priorização (kendall_tau, spearman_r)
            # não estão disponíveis — PROMISE não tem ground truth de prioridade.
            "prioritization_correlation": None,
        },
        # ── Predições brutas ──────────────────────────────────────────────────
        "predictions": [
            {
                "id":                        r.id,
                "text":                      r.text[:120],
                "ground_truth_type":         r.metadata.get("label_type"),
                "ground_truth_category":     r.metadata.get("label_category"),
                "predicted_type":            r.requirement_type.value if r.requirement_type else None,
                "predicted_category":        r.nfr_category.value if r.nfr_category else None,
                "confidence":                r.confidence,
                "classification_justification": r.justification,
                "priority":                  r.priority.value if r.priority else None,
                "priority_score":            r.priority_score,
                "priority_rank":             r.priority_rank,
                "priority_justification":    r.justification_priority,
            }
            for r in prioritized
        ],
    }

    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Resultado salvo em: %s", out_file)
    print(f"\nJSON salvo em: {out_file}\n")


if __name__ == "__main__":
    main()
