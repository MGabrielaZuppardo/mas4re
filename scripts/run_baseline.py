"""
Script de execução do BaselineAgent sobre o dataset PROMISE NFR+.

Uso básico (amostra de 10 requisitos, modelo Groq):
    python scripts/run_baseline.py

Exemplos:
    python scripts/run_baseline.py --n 20 --model llama-3.3-70b-versatile
    python scripts/run_baseline.py --full --model claude-haiku-4-5
    python scripts/run_baseline.py --n 5  --model ollama/qwen2.5:7b --lang en
    python scripts/run_baseline.py --n 10 --no-taxonomy
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
logger = logging.getLogger("run_baseline")

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.baseline import BaselineAgent
from config.settings import settings
from datasets.promise import PromiseAdapter
from domain.enums import NFRCategory
from domain.models import PipelineState
from evaluation.metrics.classification import (
    compute_classification_metrics,
    compute_subcategory_metrics,
)
from evaluation.metrics.prioritization import compute_moscow_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o BaselineAgent no PROMISE NFR+")
    parser.add_argument(
        "--model",
        default=settings.model_groq_default,
        help="Modelo LLM (ex: llama-3.3-70b-versatile, claude-haiku-4-5, ollama/qwen2.5:7b)",
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
        "--lang",
        default="pt",
        choices=["pt", "en"],
        help="Idioma do prompt do agente",
    )
    parser.add_argument(
        "--no-taxonomy",
        action="store_true",
        help="Roda sem injetar a taxonomia PROMISE no prompt (modo genérico)",
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

    # ── Categorias NFR ────────────────────────────────────────────────────────
    nfr_categories: list[tuple[str, str]] | None = None
    if not args.no_taxonomy:
        nfr_categories = [
            (c.value, c.name.replace("_", " ").title())
            for c in NFRCategory.nfr_only()
        ]
        logger.info("Taxonomia PROMISE injetada | categorias=%d", len(nfr_categories))
    else:
        logger.info("Modo genérico — sem taxonomia no prompt")

    # ── Agente ────────────────────────────────────────────────────────────────
    agent = BaselineAgent(
        model=args.model,
        nfr_categories=nfr_categories,
        lang=args.lang,
    )

    # ── Execução ──────────────────────────────────────────────────────────────
    state = PipelineState(raw_requirements=requirements)
    logger.info("Iniciando execução...")
    started_at = datetime.now()
    result = agent.run(state)
    elapsed = (datetime.now() - started_at).total_seconds()

    predictions = result.prioritized_requirements

    # ── Métricas de classificação (ground truth disponível no PROMISE) ────────
    clf_metrics = compute_classification_metrics(predictions, requirements)
    subcat_metrics = compute_subcategory_metrics(predictions, requirements)
    moscow_dist = compute_moscow_distribution(predictions)

    # ── Exibe no console ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  RESULTADOS — BaselineAgent | {args.model} | n={len(predictions)}")
    print("=" * 70)

    print("\n── Predições ──────────────────────────────────────────────────────")
    for req in predictions:
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

    print(f"\n  Tempo de execução: {elapsed:.1f}s")
    print("=" * 70)

    # ── Salva JSON ────────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode      = "full" if args.full else f"sample{args.n}"
    out_file  = out_dir / f"baseline_{args.model.replace('/', '-')}_{mode}_{timestamp}.json"

    payload = {
        "run_id":             result.run_id,
        "model":              args.model,
        "lang":               args.lang,
        "taxonomy_injected":  not args.no_taxonomy,
        "dataset":            "PROMISE_NFR_PT",
        "n":                  len(predictions),
        "elapsed_seconds":    round(elapsed, 2),
        # ── Métricas ──────────────────────────────────────────────────────────
        "metrics": {
            "classification":   clf_metrics,
            "subcategory":      subcat_metrics,
            "moscow_distribution": moscow_dist,
        },
        # ── Predições brutas ──────────────────────────────────────────────────
        "predictions": [
            {
                "id":                           r.id,
                "text":                         r.text[:120],
                "ground_truth_type":            r.metadata.get("label_type"),
                "ground_truth_category":        r.metadata.get("label_category"),
                "predicted_type":               r.requirement_type.value if r.requirement_type else None,
                "predicted_category":           r.nfr_category.value if r.nfr_category else None,
                "confidence":                   r.confidence,
                "classification_justification": r.justification,
                "priority":                     r.priority.value if r.priority else None,
                "priority_score":               r.priority_score,
                "priority_rank":                r.priority_rank,
                "priority_justification":       r.justification_priority,
            }
            for r in predictions
        ],
    }

    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Resultado salvo em: %s", out_file)
    print(f"\nJSON salvo em: {out_file}\n")


if __name__ == "__main__":
    main()
