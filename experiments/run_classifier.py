"""
MAS4RE — Experimento: ClassificationAgent isolado no PROMISE NFR+

Roda o ClassificationAgent com cada modelo configurado e compara as métricas.

Uso:
    # Todos os modelos, amostra de 50 requisitos
    docker compose run app python -m experiments.run_classifier --n 50

    # Modelo específico, dataset completo
    docker compose run app python -m experiments.run_classifier --model ollama/qwen2.5:7b

    # Todos os modelos, dataset completo
    docker compose run app python -m experiments.run_classifier --all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Garante que o root do projeto está no path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.reporter import compare_models, print_summary, save_results
from evaluation.runner import run_classifier_on_promise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Modelos disponíveis para avaliação
MODELS = [
    "ollama/qwen2.5:7b",
    "ollama/llama3.1:8b",
    "ollama/phi3.5:3.8b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia ClassificationAgent isolado no PROMISE NFR+"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo específico (ex: ollama/qwen2.5:7b). Omitir = todos.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Número de amostras. Omitir = dataset completo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade (default: 42).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Paralelismo do classify_batch (default: 3).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag para identificar o experimento nos resultados.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Roda todos os modelos definidos em MODELS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models_to_run = MODELS if args.all or args.model is None else [args.model]

    logger.info(
        "Experimento iniciado | modelos=%s | n=%s | seed=%d",
        models_to_run,
        args.n or "all",
        args.seed,
    )

    results = []

    for model in models_to_run:
        print(f"\n>>> Avaliando modelo: {model}")
        try:
            result = run_classifier_on_promise(
                model=model,
                n_samples=args.n,
                seed=args.seed,
                max_workers=args.workers,
            )
            print_summary(result)
            path = save_results(result, tag=args.tag)
            print(f"    Resultado salvo em: {path}")
            results.append(result)

        except Exception as e:
            logger.error("Falha no modelo %s | erro=%s", model, e)
            print(f"    ERRO: {e}")

    # Comparação final se rodou mais de um modelo
    if len(results) > 1:
        compare_models(results)

    logger.info("Experimento concluído | modelos_avaliados=%d", len(results))


if __name__ == "__main__":
    main()
