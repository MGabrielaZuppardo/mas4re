"""
MAS4RE — Suplemento gpt-4.1-mini (Azure AI Foundry): baseline e pipeline

As rodadas EN de baseline e pipeline do suplemento foram executadas com um código em
que o classificador e o baseline enviavam o texto PT ao modelo (com prompt EN). Este
script as reexecuta com os agentes da a13f2d6, que usam o texto original em inglês nas
condições EN (ver tests/unit/test_requirement_language_input.py). O braço two-call EN
(setembro) e os braços PT já usavam o texto correto.

Uso (a partir da raiz do repositório):
    python -m experiments.run_foundry_grid --langs en --resume
    python -m experiments.run_foundry_grid --n 3 --langs en --strategies baseline \
        --out experiments/results/_smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.enums import Lang
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import BaselineStrategy, PipelineStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiments/foundry_grid.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

MODEL = "foundry/gpt-4.1-mini"
STRATEGIES = ("baseline", "pipeline")
DATASET_PATH = "datasets/data/promise_nfr/promise_nfr_pt.csv"
TEMPERATURE = 0.0
SEED = 42
DEFAULT_OUT = "experiments/results/foundry_en_rerun"


def _run_model_label(strategy_name: str) -> str:
    return f"{MODEL}-{MODEL}" if strategy_name == "pipeline" else MODEL


def _build_strategy(strategy_name: str, lang: Lang) -> BaselineStrategy | PipelineStrategy:
    if strategy_name == "pipeline":
        return PipelineStrategy(
            classifier_model=MODEL, prioritizer_model=MODEL, temperature=TEMPERATURE, lang=lang
        )
    return BaselineStrategy(model=MODEL, temperature=TEMPERATURE, lang=lang)


def _already_done(out_dir: Path, strategy_name: str, lang: Lang) -> bool:
    slug = _run_model_label(strategy_name).replace("/", "-").replace(":", "-")
    return any(
        (run_dir / "results.json").exists() and (run_dir / "results.json").stat().st_size > 1000
        for run_dir in out_dir.glob(f"{strategy_name}_{slug}_{lang.value}_n*")
    )


def _run_condition(strategy_name: str, lang: Lang, n_samples: int | None, out_dir: Path) -> None:
    config = RunConfig(
        strategy_name=strategy_name,
        model=_run_model_label(strategy_name),
        dataset_path=DATASET_PATH,
        lang=lang.value,
        n_samples=n_samples,
        seed=SEED,
        temperature=TEMPERATURE,
    )
    result = ExperimentRunner(out_dir=str(out_dir)).execute(
        _build_strategy(strategy_name, lang), config
    )
    classification = result.metrics.get("classification", {})
    logger.info(
        "OK | %s %s | %.0fs | acc=%.4f F1=%.4f",
        strategy_name,
        lang.value,
        result.elapsed_seconds,
        classification.get("accuracy", 0.0),
        classification.get("f1_macro", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Suplemento foundry: baseline e pipeline")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=None, help="Amostras (None = dataset completo)")
    parser.add_argument("--langs", nargs="+", choices=["pt", "en"], default=["en"])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--resume", action="store_true", help="Pula condições já concluídas")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for lang in (Lang(value) for value in args.langs):
        for strategy_name in args.strategies:
            label = f"{strategy_name} {MODEL} {lang.value}"
            if args.resume and _already_done(out_dir, strategy_name, lang):
                logger.info("%s SKIP (já concluída)", label)
                continue
            logger.info("%s INICIANDO", label)
            try:
                _run_condition(strategy_name, lang, args.n, out_dir)
            except Exception as error:
                logger.error("%s ERRO | %s", label, error)
    logger.info("Concluído em %.1f min", (time.time() - started) / 60)


if __name__ == "__main__":
    main()
