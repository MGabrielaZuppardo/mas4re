"""
MAS4RE — Grid completo n=625 (dataset PROMISE NFR+ completo)

Roda 12 condições sequencialmente:
    3 modelos × 2 idiomas × 2 arquiteturas

Uso (a partir de D:/mas4re):
    python -m experiments.run_grid_full

Flags opcionais:
    --n        Número de amostras (padrão: None = dataset completo n=625)
    --seed     Seed de amostragem (padrão: 42)
    --out      Diretório de resultados (padrão: experiments/results)
    --resume   Pula condições já executadas com sucesso
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging import configure_logging
from domain.enums import Lang
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import BaselineStrategy, PipelineStrategy

logger = logging.getLogger(__name__)

# ── Configuração do grid ───────────────────────────────────────────────────────

MODELS = [
    "ollama/qwen2.5:7b",
    "ollama/llama3.1:8b",
    "ollama/mistral:7b",
]

LANGS = [Lang.PT, Lang.EN]

DATASET_PATH = "datasets/data/promise_nfr/promise_nfr_pt.csv"

TEMPERATURE = 0.0
SEED = 42


# ── Estrutura de resultado por condição ───────────────────────────────────────


@dataclass
class ConditionResult:
    strategy: str
    model: str
    lang: str
    n: int
    elapsed_seconds: float
    accuracy: float | None
    f1_macro: float | None
    mcc: float | None
    status: str
    error: str = ""

    def run_id(self) -> str:
        model_slug = self.model.replace("/", "-").replace(":", "-")
        return f"{self.strategy}_{model_slug}_{self.lang}_n{self.n}"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _already_done(summary_path: Path, run_id: str) -> bool:
    """Verifica se um run_id já foi executado com sucesso no CSV de resumo."""
    if not summary_path.exists():
        return False
    with open(summary_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("run_id", "").startswith(run_id) and row.get("status") == "ok":
                return True
    return False


def _append_csv(summary_path: Path, result: ConditionResult, run_id_full: str) -> None:
    is_new = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "run_id",
                    "strategy",
                    "model",
                    "lang",
                    "n",
                    "elapsed_seconds",
                    "accuracy",
                    "f1_macro",
                    "mcc",
                    "status",
                    "error",
                ]
            )
        writer.writerow(
            [
                run_id_full,
                result.strategy,
                result.model,
                result.lang,
                result.n,
                round(result.elapsed_seconds, 2),
                result.accuracy,
                result.f1_macro,
                result.mcc,
                result.status,
                result.error,
            ]
        )


def _run_condition(
    strategy_name: str,
    model: str,
    lang: Lang,
    n_samples: int | None,
    out_dir: str,
) -> tuple[ConditionResult, str]:
    """Executa uma condição e retorna (resultado, run_id_completo)."""
    if strategy_name == "baseline":
        strategy = BaselineStrategy(model=model, temperature=TEMPERATURE, lang=lang)
        run_model = model
    else:
        strategy = PipelineStrategy(
            classifier_model=model,
            prioritizer_model=model,
            temperature=TEMPERATURE,
            lang=lang,
        )
        run_model = f"{model}-{model}"

    config = RunConfig(
        strategy_name=strategy_name,
        model=run_model,
        dataset_path=DATASET_PATH,
        lang=lang.value,  # M1-fix: include lang for traceability
        n_samples=n_samples,
        seed=SEED,
        temperature=TEMPERATURE,
    )

    runner = ExperimentRunner(out_dir=out_dir)

    try:
        result = runner.execute(strategy, config)
        cls = result.metrics.get("classification", {})
        n_actual = result.state.n_requirements
        cond = ConditionResult(
            strategy=strategy_name,
            model=model,
            lang=lang.value,
            n=n_actual,
            elapsed_seconds=result.elapsed_seconds,
            accuracy=cls.get("accuracy"),
            f1_macro=cls.get("f1_macro"),
            mcc=cls.get("mcc"),
            status="ok",
        )
        # M1-fix: use the run_id returned by the runner (matches actual directory name).
        return cond, result.run_id

    except Exception as e:
        logger.error("Condição falhou | %s %s %s | erro=%s", strategy_name, model, lang.value, e)
        n_actual = n_samples or 625
        cond = ConditionResult(
            strategy=strategy_name,
            model=model,
            lang=lang.value,
            n=n_actual,
            elapsed_seconds=0.0,
            accuracy=None,
            f1_macro=None,
            mcc=None,
            status="error",
            error=str(e)[:200],
        )
        run_id_full = (
            f"_{strategy_name}_{model.replace('/', '-').replace(':', '-')}_{lang.value}_n{n_actual}"
        )
        return cond, run_id_full


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid completo MAS4RE")
    parser.add_argument(
        "--n", type=int, default=None, help="Amostras por condição (None = dataset completo)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="experiments/results")
    parser.add_argument(
        "--resume", action="store_true", help="Pula condições já executadas com sucesso"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    summary_name = f"grid_summary_n{'full' if args.n is None else args.n}_{timestamp}.csv"
    summary_path = Path(args.out) / summary_name
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Grid: 3 modelos × 2 langs × 2 estratégias = 12 condições
    conditions = [
        (strategy, model, lang)
        for model in MODELS
        for lang in LANGS
        for strategy in ["baseline", "pipeline"]
    ]

    total = len(conditions)
    n_label = args.n or 625

    logger.info("=" * 60)
    logger.info("MAS4RE Grid Full | condições=%d | n=%s | seed=%d", total, n_label, SEED)
    logger.info("Resumo: %s", summary_path)
    logger.info("=" * 60)

    # Estimar tempo total
    secs_per_req = 5.97
    est_total_h = (secs_per_req * n_label * total) / 3600
    logger.info("Tempo estimado (sequencial): ~%.1fh", est_total_h)

    results: list[ConditionResult] = []

    for i, (strategy, model, lang) in enumerate(conditions, start=1):
        prefix = f"[{i:02d}/{total}]"
        model_short = model.split("/")[-1]

        # Verificar se pode pular (--resume)
        run_id_prefix = f"{strategy}_{model.replace('/', '-').replace(':', '-')}_{lang.value}_n"
        if args.resume and _already_done(summary_path, run_id_prefix):
            logger.info(
                "%s SKIP (já executado) | %s %s %s", prefix, strategy, model_short, lang.value
            )
            continue

        logger.info(
            "%s INICIANDO | %s | %s | %s | n=%s", prefix, strategy, model_short, lang.value, n_label
        )

        t_start = time.time()
        cond_result, run_id_full = _run_condition(
            strategy_name=strategy,
            model=model,
            lang=lang,
            n_samples=args.n,
            out_dir=args.out,
        )
        elapsed = time.time() - t_start

        # Estimar tempo restante
        remaining = total - i
        est_remaining_h = (elapsed * remaining) / 3600

        if cond_result.status == "ok":
            logger.info(
                "%s OK | %.0fs | F1=%.4f ACC=%.4f MCC=%.4f | restante ~%.1fh",
                prefix,
                elapsed,
                cond_result.f1_macro or 0,
                cond_result.accuracy or 0,
                cond_result.mcc or 0,
                est_remaining_h,
            )
        else:
            logger.error(
                "%s ERRO | %s | restante ~%.1fh", prefix, cond_result.error, est_remaining_h
            )

        _append_csv(summary_path, cond_result, run_id_full)
        results.append(cond_result)

    # Sumário final
    ok = [r for r in results if r.status == "ok"]
    err = [r for r in results if r.status == "error"]

    logger.info("=" * 60)
    logger.info("GRID CONCLUÍDO | ok=%d | erros=%d | resumo=%s", len(ok), len(err), summary_path)

    if ok:
        logger.info("\nResultados (condições bem-sucedidas):")
        logger.info(
            f"  {'Estrategia':<12} {'Modelo':<20} {'Lang':<5} {'F1':>8} {'ACC':>8} {'MCC':>8}"
        )
        for r in sorted(ok, key=lambda x: (x.model, x.lang, x.strategy)):
            logger.info(
                "  %-12s %-20s %-5s %8.4f %8.4f %8.4f",
                r.strategy,
                r.model.split("/")[-1],
                r.lang,
                r.f1_macro or 0,
                r.accuracy or 0,
                r.mcc or 0,
            )

    if err:
        logger.info("\nCondições com erro:")
        for r in err:
            logger.info("  %s %s %s: %s", r.strategy, r.model, r.lang, r.error)

    logger.info("=" * 60)


if __name__ == "__main__":
    configure_logging(log_file="experiments/grid_full.log")
    main()
