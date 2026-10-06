"""
MAS4RE — Repetição das condições pipeline e two-call (variação entre execuções idênticas)

temperature=0.0 não garante decodificação bit-determinística no Ollama. Este
script reexecuta, com a mesma configuração, seed e código das rodadas originais,
as condições pipeline e two_call_baseline, gravando em um diretório separado. As
rodadas repetidas permitem medir quanto o Fleiss' kappa e o Delta-kappa variam
apenas por não-determinismo (compute_repeat_kappa_variance.py).

Ordem: idioma > estratégia > modelo (PT primeiro, que tem o menor Delta-kappa).
Cada condição leva cerca de 50-70 min no hardware do estudo.

Uso (a partir da raiz do repositório):
    python -m experiments.run_repeat_grid --resume
    python -m experiments.run_repeat_grid --langs pt --resume
    python -m experiments.run_repeat_grid --n 3 --langs en --strategies pipeline \
        --models qwen --out experiments/results/_smoke
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
from experiments.strategy import PipelineStrategy, TwoCallBaselineStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiments/repeat_grid.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

MODELS = ["ollama/qwen2.5:7b", "ollama/llama3.1:8b", "ollama/mistral:7b"]
STRATEGIES = ("pipeline", "two_call_baseline")
DATASET_PATH = "datasets/data/promise_nfr/promise_nfr_pt.csv"
TEMPERATURE = 0.0
SEED = 42
DEFAULT_OUT = "experiments/results/repeat_run2"


def _slug(text: str) -> str:
    return text.replace("/", "-").replace(":", "-")


def _run_model_label(strategy_name: str, model: str) -> str:
    return f"{model}-{model}" if strategy_name == "pipeline" else model


def _build_strategy(
    strategy_name: str, model: str, lang: Lang
) -> PipelineStrategy | TwoCallBaselineStrategy:
    if strategy_name == "pipeline":
        return PipelineStrategy(
            classifier_model=model,
            prioritizer_model=model,
            temperature=TEMPERATURE,
            lang=lang,
        )
    return TwoCallBaselineStrategy(model=model, temperature=TEMPERATURE, lang=lang)


def _already_done(out_dir: Path, strategy_name: str, model: str, lang: Lang) -> bool:
    prefix = f"{strategy_name}_{_slug(_run_model_label(strategy_name, model))}_{lang.value}_n"
    return any(
        (run_dir / "results.json").exists() and (run_dir / "results.json").stat().st_size > 1000
        for run_dir in out_dir.glob(f"{prefix}*")
    )


def _conditions(
    langs: list[Lang], strategies: list[str], models: list[str]
) -> list[tuple[Lang, str, str]]:
    return [
        (lang, strategy, model) for lang in langs for strategy in strategies for model in models
    ]


def _run_condition(
    lang: Lang, strategy_name: str, model: str, n_samples: int | None, out_dir: Path
) -> float:
    config = RunConfig(
        strategy_name=strategy_name,
        model=_run_model_label(strategy_name, model),
        dataset_path=DATASET_PATH,
        lang=lang.value,
        n_samples=n_samples,
        seed=SEED,
        temperature=TEMPERATURE,
    )
    result = ExperimentRunner(out_dir=str(out_dir)).execute(
        _build_strategy(strategy_name, model, lang), config
    )
    classification = result.metrics.get("classification", {})
    logger.info(
        "OK | %s %s %s | %.0fs | F1=%.4f",
        strategy_name,
        model,
        lang.value,
        result.elapsed_seconds,
        classification.get("f1_macro", 0.0),
    )
    return result.elapsed_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Repetição pipeline/two-call (não-determinismo)")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=None, help="Amostras (None = dataset completo)")
    parser.add_argument("--langs", nargs="+", choices=["pt", "en"], default=["pt", "en"])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--models", nargs="+", default=None, help="Filtro por trecho do nome")
    parser.add_argument("--resume", action="store_true", help="Pula condições já concluídas")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODELS if args.models is None or any(f in m for f in args.models)]
    conditions = _conditions([Lang(lang) for lang in args.langs], args.strategies, models)

    logger.info("Repetição | condições=%d | out=%s | n=%s", len(conditions), out_dir, args.n or 625)
    started = time.time()
    for index, (lang, strategy_name, model) in enumerate(conditions, start=1):
        label = f"[{index:02d}/{len(conditions)}] {strategy_name} {model} {lang.value}"
        if args.resume and _already_done(out_dir, strategy_name, model, lang):
            logger.info("%s SKIP (já concluída)", label)
            continue
        logger.info("%s INICIANDO", label)
        try:
            _run_condition(lang, strategy_name, model, args.n, out_dir)
        except Exception as error:
            logger.error("%s ERRO | %s", label, error)
    logger.info("Repetição concluída em %.1f h", (time.time() - started) / 3600)


if __name__ == "__main__":
    main()
