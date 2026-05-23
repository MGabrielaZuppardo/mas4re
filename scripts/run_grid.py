"""Grid runner — executa as 12 condições experimentais do SBCARS 2026.

Condições: 3 modelos × 2 idiomas × 2 arquiteturas = 12 runs.

Uso:
    python scripts/run_grid.py              # grid completo
    python scripts/run_grid.py --n 20       # amostra de 20 requisitos (dry-run)
    python scripts/run_grid.py --dry-run    # imprime condições sem executar

Artefatos gerados por run:
    experiments/results/{run_id}/manifest.json
    experiments/results/{run_id}/results.json
    experiments/traces/{run_id}.jsonl

Sumário final:
    experiments/results/grid_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path when called as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from domain.enums import Lang
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import BaselineStrategy, PipelineStrategy

# ---------------------------------------------------------------------------
# Grid definition — 3 models × 2 languages × 2 architectures = 12 conditions
# ---------------------------------------------------------------------------

MODELS = [
    "ollama/qwen2.5:7b",
    "ollama/llama3.1:8b",
    "ollama/mistral:7b",
]

LANGUAGES = ["pt", "en"]

SUMMARY_PATH = Path("experiments/results/grid_summary.csv")  # default, overridden by --output
SUMMARY_FIELDS = [
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


@dataclass
class GridCondition:
    model: str
    lang: str
    strategy: str  # "baseline" | "pipeline"


def _build_conditions() -> list[GridCondition]:
    conditions = []
    for model in MODELS:
        for lang in LANGUAGES:
            conditions.append(GridCondition(model=model, lang=lang, strategy="baseline"))
            conditions.append(GridCondition(model=model, lang=lang, strategy="pipeline"))
    return conditions


def _build_strategy(cond: GridCondition) -> BaselineStrategy | PipelineStrategy:
    lang_enum = Lang(cond.lang)
    if cond.strategy == "baseline":
        return BaselineStrategy(model=cond.model, lang=lang_enum)
    return PipelineStrategy(
        classifier_model=cond.model,
        prioritizer_model=cond.model,
        lang=lang_enum,
    )


def _build_config(cond: GridCondition, n: int | None) -> RunConfig:
    model_label = cond.model if cond.strategy == "baseline" else f"{cond.model}+{cond.model}"
    return RunConfig(
        strategy_name=cond.strategy,
        model=model_label,
        dataset_path=settings.promise_dataset_path,
        lang=cond.lang,
        n_samples=n,
        seed=42,
    )


def _append_summary(row: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_PATH.exists()
    with SUMMARY_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_grid(n: int | None = None, dry_run: bool = False, output: Path | None = None) -> None:
    global SUMMARY_PATH
    if output is not None:
        SUMMARY_PATH = output
    conditions = _build_conditions()
    runner = ExperimentRunner()
    total = len(conditions)

    print(f"\nMAS4RE Grid — {total} condições | n={'full' if n is None else n} | seed=42\n")

    for i, cond in enumerate(conditions, start=1):
        label = f"[{i:02d}/{total}] {cond.strategy:<10} {cond.model:<25} lang={cond.lang}"
        print(label, end=" ... ", flush=True)

        if dry_run:
            print("(dry-run)")
            continue

        strategy = _build_strategy(cond)
        config = _build_config(cond, n)
        t0 = time.monotonic()

        try:
            result = runner.execute(strategy, config)
            elapsed = time.monotonic() - t0
            cls = result.metrics.get("classification", {})
            row = {
                "run_id": result.run_id,
                "strategy": cond.strategy,
                "model": cond.model,
                "lang": cond.lang,
                "n": result.manifest.get("dataset_n", ""),
                "elapsed_seconds": round(elapsed, 2),
                "accuracy": cls.get("accuracy", ""),
                "f1_macro": cls.get("f1_macro", ""),
                "mcc": cls.get("mcc", ""),
                "status": "ok",
                "error": "",
            }
            print(f"ok  ({elapsed:.1f}s | acc={cls.get('accuracy', '?'):.3f})")
        except Exception as e:
            elapsed = time.monotonic() - t0
            row = {
                "run_id": "",
                "strategy": cond.strategy,
                "model": cond.model,
                "lang": cond.lang,
                "n": n or "",
                "elapsed_seconds": round(elapsed, 2),
                "accuracy": "",
                "f1_macro": "",
                "mcc": "",
                "status": "error",
                "error": str(e),
            }
            print(f"ERRO ({e})")
            traceback.print_exc()

        _append_summary(row)

    if not dry_run:
        print(f"\nSumário salvo em: {SUMMARY_PATH}")
        _print_summary()


def _print_summary() -> None:
    if not SUMMARY_PATH.exists():
        return
    rows = list(csv.DictReader(SUMMARY_PATH.open(encoding="utf-8")))
    ok = sum(1 for r in rows if r["status"] == "ok")
    err = sum(1 for r in rows if r["status"] == "error")
    print(f"\nResultado: {ok} ok / {err} erro(s) de {len(rows)} condições")

    if ok:
        f1s = [float(r["f1_macro"]) for r in rows if r["f1_macro"]]
        print(f"F1-macro: min={min(f1s):.3f}  max={max(f1s):.3f}  mean={sum(f1s) / len(f1s):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAS4RE Grid Runner")
    parser.add_argument("--n", type=int, default=None, help="Sample size (None = full dataset)")
    parser.add_argument("--dry-run", action="store_true", help="Print conditions without running")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path for summary")
    args = parser.parse_args()
    run_grid(n=args.n, dry_run=args.dry_run, output=args.output)
