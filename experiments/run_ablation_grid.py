"""
MAS4RE — Ablation grid: TwoCallBaseline vs Pipeline

Runs 6 conditions to isolate the contribution of the typed inter-agent
state contract from prompt specialisation:

    3 models × 2 languages × 1 strategy (two_call_baseline) = 6 conditions

Then compares against the already-executed pipeline results from the main
grid (experiments/results/).  Statistical tests mirror the RQ1 protocol
(Wilcoxon signed-rank + Cohen's h + Bonferroni α' = 0.05/6 = 0.0083).

Usage (from D:/mas4re):
    python -m experiments.run_ablation_grid
    python -m experiments.run_ablation_grid --resume
    python -m experiments.run_ablation_grid --n 50   # quick smoke-test

Output:
    experiments/results/ablation_<timestamp>/
        <run_id>/results.json          # per-condition artifacts
    experiments/results/ablation_<timestamp>/ablation_summary.csv
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
from experiments.strategy import TwoCallBaselineStrategy

logger = logging.getLogger(__name__)

# ── Grid configuration ─────────────────────────────────────────────────────────

MODELS = [
    "ollama/qwen2.5:7b",
    "ollama/llama3.1:8b",
    "ollama/mistral:7b",
]

LANGS = [Lang.PT, Lang.EN]

DATASET_PATH = "datasets/data/promise_nfr/promise_nfr_pt.csv"
TEMPERATURE = 0.0
SEED = 42


# ── Result container ───────────────────────────────────────────────────────────


@dataclass
class AblationConditionResult:
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

    def run_id_prefix(self) -> str:
        model_slug = self.model.replace("/", "-").replace(":", "-")
        return f"{self.strategy}_{model_slug}_{self.lang}_n"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _already_done(summary_path: Path, prefix: str) -> bool:
    if not summary_path.exists():
        return False
    with open(summary_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("run_id", "").startswith(prefix) and row.get("status") == "ok":
                return True
    return False


def _append_csv(summary_path: Path, result: AblationConditionResult, run_id: str) -> None:
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
                run_id,
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
    model: str,
    lang: Lang,
    n_samples: int | None,
    out_dir: str,
) -> tuple[AblationConditionResult, str]:
    strategy = TwoCallBaselineStrategy(
        model=model,
        temperature=TEMPERATURE,
        lang=lang,
    )
    config = RunConfig(
        strategy_name="two_call_baseline",
        model=model,
        dataset_path=DATASET_PATH,
        lang=lang.value,
        n_samples=n_samples,
        seed=SEED,
        temperature=TEMPERATURE,
    )
    runner = ExperimentRunner(out_dir=out_dir)
    try:
        result = runner.execute(strategy, config)
        cls = result.metrics.get("classification", {})
        cond = AblationConditionResult(
            strategy="two_call_baseline",
            model=model,
            lang=lang.value,
            n=result.state.n_requirements,
            elapsed_seconds=result.elapsed_seconds,
            accuracy=cls.get("accuracy"),
            f1_macro=cls.get("f1_macro"),
            mcc=cls.get("mcc"),
            status="ok",
        )
        return cond, result.run_id
    except Exception as e:
        logger.error("Condição falhou | %s %s | erro=%s", model, lang.value, e)
        n_actual = n_samples or 625
        cond = AblationConditionResult(
            strategy="two_call_baseline",
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
        slug = model.replace("/", "-").replace(":", "-")
        return cond, f"two_call_baseline_{slug}_{lang.value}_n{n_actual}"


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MAS4RE Ablation Grid")
    parser.add_argument(
        "--n", type=int, default=None, help="Samples per condition (None = full dataset, n=625)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="experiments/results")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed conditions")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    ablation_dir = Path(args.out) / f"ablation_{timestamp}"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ablation_dir / "ablation_summary.csv"

    conditions = [(model, lang) for model in MODELS for lang in LANGS]
    total = len(conditions)
    n_label = args.n or 625

    logger.info("=" * 60)
    logger.info("MAS4RE Ablation Grid | strategy=two_call_baseline")
    logger.info("Condições: %d | n=%s | seed=%d", total, n_label, SEED)
    logger.info("Saída: %s", ablation_dir)
    logger.info("=" * 60)
    logger.info("INTERPRETAÇÃO: compare two_call_baseline vs pipeline.")
    logger.info("  Se ΔF1(pipeline - two_call_baseline) > 0 → typed state contribui.")
    logger.info("  Se ΔF1 ≈ 0 → ganho vem dos prompts especializados, não do estado.")
    logger.info("=" * 60)

    # Estimate runtime: two LLM calls per requirement vs one for baseline
    secs_per_req = 5.97 * 2  # ~2× baseline (two calls per req)
    est_total_h = (secs_per_req * n_label * total) / 3600
    logger.info(
        "Tempo estimado: ~%.1fh (2 calls/req × %d reqs × %d condições)", est_total_h, n_label, total
    )

    results: list[AblationConditionResult] = []

    for i, (model, lang) in enumerate(conditions, start=1):
        prefix = f"[{i:02d}/{total}]"
        model_short = model.split("/")[-1]
        run_id_prefix = (
            f"two_call_baseline_{model.replace('/', '-').replace(':', '-')}_{lang.value}_n"
        )

        if args.resume and _already_done(summary_path, run_id_prefix):
            logger.info("%s SKIP | %s %s", prefix, model_short, lang.value)
            continue

        logger.info("%s INICIANDO | %s | %s | n=%s", prefix, model_short, lang.value, n_label)

        t_start = time.time()
        cond_result, run_id_full = _run_condition(
            model=model,
            lang=lang,
            n_samples=args.n,
            out_dir=str(ablation_dir),
        )
        elapsed = time.time() - t_start

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

    # ── Summary ───────────────────────────────────────────────────────────────
    ok = [r for r in results if r.status == "ok"]
    err = [r for r in results if r.status == "error"]

    logger.info("=" * 60)
    logger.info("ABLATION CONCLUÍDA | ok=%d | erros=%d", len(ok), len(err))

    if ok:
        logger.info("\nResultados two_call_baseline:")
        logger.info(f"  {'Modelo':<20} {'Lang':<5} {'F1':>8} {'ACC':>8} {'MCC':>8}")
        for r in sorted(ok, key=lambda x: (x.model, x.lang)):
            logger.info(
                "  %-20s %-5s %8.4f %8.4f %8.4f",
                r.model.split("/")[-1],
                r.lang,
                r.f1_macro or 0,
                r.accuracy or 0,
                r.mcc or 0,
            )
        logger.info(
            "\nPróximo passo: rode experiments/compute_ablation_stats.py "
            "--ablation %s --grid <grid_summary.csv>",
            summary_path,
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    configure_logging(log_file="experiments/ablation_grid.log")
    main()
