"""How much room does Condition B have to act? (historical runs, no LLM calls)

Two quantities bound what a mediated coordination can change:
  * the number of cross_check conflicts per run (the ceiling of the effect on accuracy is
    conflicts / n), and
  * the share of Must-Have priorities per run (a batch-level inflation signal, checked
    against the DSDM guideline and against the integration-test limit).

Usage (from the repository root):
    python -m experiments.analyze_coordination_power
    python -m experiments.analyze_coordination_power --must-limit 0.60 --strict-limit 0.80

Output:
    experiments/results/analysis/coordination_power_<timestamp>.json
"""

from __future__ import annotations

import argparse
from typing import Any

from experiments.analysis_common import (
    FULL_DATASET_N,
    GENERATIONS,
    HISTORICAL,
    ZERO_SHOT,
    Prediction,
    RunData,
    build_analysis_manifest,
    has_inter_agent_conflict,
    is_critical_nfr,
    load_full_runs,
    write_analysis_result,
)

DSDM_MUST_LIMIT = 0.60
INTEGRATION_TEST_MUST_LIMIT = 0.80


def conflict_ceiling(predictions: list[Prediction]) -> dict[str, float | int]:
    conflicts = sum(map(has_inter_agent_conflict, predictions))
    return {
        "conflicts": conflicts,
        "critical_items": sum(map(is_critical_nfr, predictions)),
        "effect_ceiling": conflicts / len(predictions) if predictions else 0.0,
    }


def must_share(run: RunData) -> float | None:
    return run.metrics.get("moscow_distribution", {}).get("M")


def run_row(run: RunData) -> dict[str, Any]:
    row: dict[str, Any] = {
        "strategy": run.strategy,
        "model": run.model,
        "lang": run.lang,
        "must_share": must_share(run),
    }
    if run.strategy == "pipeline":
        row.update(conflict_ceiling(run.predictions))
    return row


def summarize(rows: list[dict[str, Any]], must_limit: float, strict_limit: float) -> dict[str, Any]:
    pipeline_rows = [r for r in rows if "conflicts" in r]
    with_must = [r for r in rows if r["must_share"] is not None]
    return {
        "pipeline_runs": len(pipeline_rows),
        "pipeline_runs_with_conflict": sum(1 for r in pipeline_rows if r["conflicts"] > 0),
        "total_conflicts": sum(r["conflicts"] for r in pipeline_rows),
        "max_conflicts_in_a_run": max((r["conflicts"] for r in pipeline_rows), default=0),
        "runs_with_must_share": len(with_must),
        f"runs_over_{must_limit:.2f}": sum(1 for r in with_must if r["must_share"] > must_limit),
        f"runs_over_{strict_limit:.2f}": sum(
            1 for r in with_must if r["must_share"] > strict_limit
        ),
    }


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print(f"{'strategy':<20}{'model':<24}{'lang':<6}{'must':>7}{'conflicts':>11}{'ceiling':>9}")
    for r in sorted(rows, key=lambda r: (r["strategy"], r["model"], str(r["lang"]))):
        must = "n/a" if r["must_share"] is None else f"{r['must_share']:.0%}"
        conflicts = r.get("conflicts", "")
        ceiling = f"{r['effect_ceiling']:.2%}" if "effect_ceiling" in r else ""
        print(
            f"{r['strategy']:<20}{r['model']:<24}{str(r['lang']):<6}{must:>7}"
            f"{conflicts!s:>11}{ceiling:>9}"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=FULL_DATASET_N)
    parser.add_argument("--generation", choices=list(GENERATIONS), default=HISTORICAL)
    parser.add_argument("--must-limit", type=float, default=DSDM_MUST_LIMIT)
    parser.add_argument("--strict-limit", type=float, default=INTEGRATION_TEST_MUST_LIMIT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    runs = load_full_runs(n=args.n, generation=args.generation)
    rows = [run_row(run) for run in runs]
    summary = summarize(rows, args.must_limit, args.strict_limit)

    _print_rows(rows)
    print(f"\n{summary}")

    manifest = build_analysis_manifest(
        "analyze_coordination_power",
        ZERO_SHOT,
        {
            "n": args.n,
            "generation": args.generation,
            "must_limit": args.must_limit,
            "strict_limit": args.strict_limit,
            "runs_loaded": len(runs),
        },
    )
    path = write_analysis_result("coordination_power", manifest, {"runs": rows, "summary": summary})
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
