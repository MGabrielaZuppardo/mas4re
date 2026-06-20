"""
MAS4RE — Rerun seletivo: pipeline mistral:7b PT e EN com parser fix.

Roda apenas 2 condições (das 12 do grid original) para validar o
schema-compliance fix em classifier._parse_response.

Uso (a partir de D:/mas4re):
    python experiments/run_mistral_pipeline_fix.py

Saída: novo CSV  experiments/results/mistral_fix_<timestamp>.csv
       novos dirs experiments/results/pipeline_ollama-mistral-7b-*
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.enums import Lang
from experiments.run_grid_full import ConditionResult, _run_condition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiments/mistral_fix.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

OUT_DIR = "experiments/results"
MODEL = "ollama/mistral:7b"
CONDITIONS = [
    ("pipeline", MODEL, Lang.PT),
    ("pipeline", MODEL, Lang.EN),
]


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    summary_path = Path(OUT_DIR) / f"mistral_fix_{timestamp}.csv"

    logger.info("=" * 60)
    logger.info("MAS4RE — Mistral Pipeline Fix Rerun")
    logger.info("Condições: %d  |  modelo: %s", len(CONDITIONS), MODEL)
    logger.info("Saída: %s", summary_path)
    logger.info("Tempo estimado: ~2h30")
    logger.info("=" * 60)

    results: list[tuple[ConditionResult, str]] = []

    for i, (strategy, model, lang) in enumerate(CONDITIONS, start=1):
        label = f"[{i}/{len(CONDITIONS)}] {strategy} {model.split('/')[-1]} {lang.value.upper()}"
        logger.info("%s — iniciando...", label)

        t0 = time.time()
        cond, run_id = _run_condition(  # run_id now comes from result.run_id (M1-fix)
            strategy_name=strategy,
            model=model,
            lang=lang,
            n_samples=None,  # dataset completo n=625
            out_dir=OUT_DIR,
        )
        elapsed = time.time() - t0

        if cond.status == "ok":
            logger.info(
                "%s — OK | %.0fs | F1=%.4f  ACC=%.4f  MCC=%.4f",
                label,
                elapsed,
                cond.f1_macro or 0,
                cond.accuracy or 0,
                cond.mcc or 0,
            )
        else:
            logger.error("%s — ERRO: %s", label, cond.error)

        results.append((cond, run_id))

    # ── Salvar CSV de resumo ──────────────────────────────────────────────────
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
        for cond, run_id in results:
            writer.writerow(
                [
                    run_id,
                    cond.strategy,
                    cond.model,
                    cond.lang,
                    cond.n,
                    round(cond.elapsed_seconds, 2),
                    cond.accuracy,
                    cond.f1_macro,
                    cond.mcc,
                    cond.status,
                    cond.error,
                ]
            )

    # ── Sumário final ─────────────────────────────────────────────────────────
    ok = [c for c, _ in results if c.status == "ok"]
    err = [c for c, _ in results if c.status == "error"]

    logger.info("=" * 60)
    logger.info("CONCLUÍDO | ok=%d | erros=%d", len(ok), len(err))
    logger.info("")
    logger.info("  Resultados (comparar com originais):")
    logger.info("  %-12s %-6s  %8s  %8s  %8s", "Estrategia", "Lang", "F1", "ACC", "MCC")
    logger.info("  " + "-" * 50)

    originals = {
        "pt": {"f1": 0.5221, "acc": 0.5504, "mcc": 0.3073},
        "en": {"f1": 0.6073, "acc": 0.6192, "mcc": 0.4045},
    }
    for cond in ok:
        orig = originals[cond.lang]
        delta_f1 = (cond.f1_macro or 0) - orig["f1"]
        delta_acc = (cond.accuracy or 0) - orig["acc"]
        logger.info(
            "  %-12s %-6s  %8.4f  %8.4f  %8.4f   (Df1=%+.4f  Dacc=%+.4f)",
            cond.strategy,
            cond.lang.upper(),
            cond.f1_macro or 0,
            cond.accuracy or 0,
            cond.mcc or 0,
            delta_f1,
            delta_acc,
        )
    if err:
        for cond in err:
            logger.error("  ERRO %s %s: %s", cond.strategy, cond.lang, cond.error)

    logger.info("=" * 60)
    logger.info("Resumo salvo: %s", summary_path)


if __name__ == "__main__":
    main()
