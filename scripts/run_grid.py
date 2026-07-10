"""Grid runner — executa as 18 condições experimentais do MAS4RE.

Condições: 3 modelos × 2 idiomas × 3 arquiteturas = 18 runs
(baseline, pipeline, two_call_baseline — esta última isola a
contribuição do contrato de estado tipado por ablação, ver
experiments/strategy.py::TwoCallBaselineStrategy).

Uso:
    python scripts/run_grid.py              # grid completo
    python scripts/run_grid.py --n 20       # amostra de 20 requisitos
    python scripts/run_grid.py --dry-run    # imprime condições sem executar
    python scripts/run_grid.py --resume     # pula condições já concluídas (status=ok no CSV)
    python scripts/run_grid.py --workers 4  # paraleliza N condições simultaneamente

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
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path when called as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from domain.enums import Lang
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import BaselineStrategy, PipelineStrategy, TwoCallBaselineStrategy

# ---------------------------------------------------------------------------
# Grid definition — 3 models × 2 languages × 3 architectures = 18 conditions
# ---------------------------------------------------------------------------

# Permite sobrescrever modelos via GRID_MODELS (separados por vírgula)
# Exemplo: GRID_MODELS=azure/gpt-5-nano,azure/gpt-5,azure/gpt-5-chat
_env_models = os.environ.get("GRID_MODELS", "")
MODELS = (
    [m.strip() for m in _env_models.split(",") if m.strip()]
    if _env_models
    else [
        "ollama/qwen2.5:7b",
        "ollama/llama3.1:8b",
        "ollama/mistral:7b",
    ]
)

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
    strategy: str  # "baseline" | "pipeline" | "two_call_baseline"


def _build_conditions() -> list[GridCondition]:
    conditions = []
    for model in MODELS:
        for lang in LANGUAGES:
            conditions.append(GridCondition(model=model, lang=lang, strategy="baseline"))
            conditions.append(GridCondition(model=model, lang=lang, strategy="pipeline"))
            conditions.append(GridCondition(model=model, lang=lang, strategy="two_call_baseline"))
    return conditions


def _build_strategy(
    cond: GridCondition,
) -> BaselineStrategy | PipelineStrategy | TwoCallBaselineStrategy:
    lang_enum = Lang(cond.lang)
    if cond.strategy == "baseline":
        return BaselineStrategy(model=cond.model, lang=lang_enum)
    if cond.strategy == "two_call_baseline":
        return TwoCallBaselineStrategy(model=cond.model, lang=lang_enum)
    return PipelineStrategy(
        classifier_model=cond.model,
        prioritizer_model=cond.model,
        lang=lang_enum,
    )


def _build_config(cond: GridCondition, n: int | None, seed: int = 42) -> RunConfig:
    # Only "pipeline" runs two distinct agent slots (classifier + prioritizer);
    # "baseline" and "two_call_baseline" both run a single model end to end.
    model_label = cond.model if cond.strategy != "pipeline" else f"{cond.model}+{cond.model}"
    return RunConfig(
        strategy_name=cond.strategy,
        model=model_label,
        dataset_path=settings.promise_dataset_path,
        lang=cond.lang,
        n_samples=n,
        seed=seed,
    )


def _load_completed(summary_path: Path) -> set[tuple[str, str, str]]:
    """Return set of (strategy, model, lang) already completed with status=ok."""
    if not summary_path.exists():
        return set()
    completed: set[tuple[str, str, str]] = set()
    with summary_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                completed.add((row["strategy"], row["model"], row["lang"]))
    return completed


_summary_lock = threading.Lock()


def _append_summary(row: dict) -> None:
    with _summary_lock:
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not SUMMARY_PATH.exists()
        with SUMMARY_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _run_condition(
    cond: GridCondition,
    index: int,
    total: int,
    n: int | None,
    seed: int,
) -> dict:
    label = f"[{index:02d}/{total}] {cond.strategy:<10} {cond.model:<25} lang={cond.lang}"
    runner = ExperimentRunner()
    strategy = _build_strategy(cond)
    config = _build_config(cond, n, seed=seed)
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
        print(f"{label} ... ok  ({elapsed:.1f}s | acc={cls.get('accuracy', '?'):.3f})", flush=True)
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
        print(f"{label} ... ERRO ({e})", flush=True)
        traceback.print_exc()
    return row


def run_grid(
    n: int | None = None,
    dry_run: bool = False,
    output: Path | None = None,
    seed: int | None = None,
    resume: bool = False,
    workers: int = 1,
) -> None:
    import random as _random

    global SUMMARY_PATH
    if output is not None:
        SUMMARY_PATH = output

    effective_seed = seed if seed is not None else _random.randint(0, 2**31 - 1)

    conditions = _build_conditions()
    completed = _load_completed(SUMMARY_PATH) if resume else set()
    total = len(conditions)

    print(
        f"\nMAS4RE Grid — {total} condições | n={'full' if n is None else n}"
        f" | seed={effective_seed} | workers={workers}"
        + (f" | resume=on ({len(completed)} já concluídas)" if resume else "")
        + "\n"
    )

    pending = []
    for i, cond in enumerate(conditions, start=1):
        label = f"[{i:02d}/{total}] {cond.strategy:<10} {cond.model:<25} lang={cond.lang}"
        if dry_run:
            print(f"{label} ... (dry-run)")
            continue
        if resume and (cond.strategy, cond.model, cond.lang) in completed:
            print(f"{label} ... (skipped — já concluída)")
            continue
        pending.append((i, cond))

    if dry_run:
        return

    if workers <= 1:
        for i, cond in pending:
            row = _run_condition(cond, i, total, n, effective_seed)
            _append_summary(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_condition, cond, i, total, n, effective_seed): cond
                for i, cond in pending
            }
            for future in as_completed(futures):
                row = future.result()
                _append_summary(row)

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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed (omit for random seed, use 42 for reproducibility)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip conditions already completed with status=ok in the summary CSV",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Número de condições paralelas (default=1 = serial). Use 3-4 com API em cloud.",
    )
    args = parser.parse_args()
    run_grid(
        n=args.n,
        dry_run=args.dry_run,
        output=args.output,
        seed=args.seed,
        resume=args.resume,
        workers=args.workers,
    )
