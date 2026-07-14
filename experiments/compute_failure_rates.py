"""
MAS4RE — Métricas de taxa de falha agregadas por modo/estágio/arquitetura (ADR-003)

Lê os `failure_records` já persistidos em cada experiments/results/{run_id}/results.json
(gerados pela DetectorChain, ligada nos 4 agentes, mais os `inter_agent_conflicts`
do cross_check_node para a arquitetura pipeline) e agrega, por
(strategy, model, lang):

- failure_rate:  fração de itens excluídos das métricas de qualidade
                 (severidade FATAL — ex.: schema_invalid / parse falhou)
- flagged_rate:  fração de itens mantidos mas sinalizados
                 (severidade DEGRADED/FLAGGED — ex.: baixa confiança,
                 categoria alucinada, justificativa sem grounding)
- contagem por (stage, mode, severity)
- inter_agent_conflict_rate (só pipeline, via cross_check_node)

Não executa nenhum experimento — só agrega resultados já coletados.

Uso:
    python experiments/compute_failure_rates.py
    python experiments/compute_failure_rates.py --results-dir experiments/results \\
        --output experiments/results/failure_rates.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.json"
    if not manifest_path.exists() or not results_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return {"manifest": manifest, "results": results}


def compute_failure_rates(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run = _load_run(run_dir)
        if run is None:
            continue

        manifest = run["manifest"]
        results = run["results"]
        n_dataset = manifest.get("dataset_n", 0)
        if not n_dataset:
            continue

        records = results.get("failure_records", [])
        conflicts = results.get("metrics", {}).get("inter_agent_conflicts", [])

        fatal_ids: set[str] = set()
        flagged_ids: set[str] = set()
        by_stage_mode: Counter[tuple[str, str, str]] = Counter()
        for r in records:
            by_stage_mode[(r["stage"], r["mode"], r["severity"])] += 1
            if r["severity"] == "fatal":
                fatal_ids.add(r["requirement_id"])
            else:
                flagged_ids.add(r["requirement_id"])
        # An item flagged in one stage and fatal in another still counts as fatal
        # for the exclusion-from-metrics semantics (BatchResult: fatal wins).
        flagged_ids -= fatal_ids

        rows.append(
            {
                "run_id": manifest.get("run_id", run_dir.name),
                "strategy": manifest.get("strategy"),
                "model": manifest.get("model"),
                "lang": manifest.get("lang"),
                "n_dataset": n_dataset,
                "n_predictions": results.get("n_predictions"),
                "n_fatal_items": len(fatal_ids),
                "n_flagged_items": len(flagged_ids),
                "failure_rate": round(len(fatal_ids) / n_dataset, 4),
                "flagged_rate": round(len(flagged_ids) / n_dataset, 4),
                "n_inter_agent_conflicts": len(conflicts),
                "inter_agent_conflict_rate": round(len(conflicts) / n_dataset, 4)
                if conflicts or manifest.get("strategy") == "pipeline"
                else None,
                "by_stage_mode_severity": {
                    f"{stage}/{mode}/{severity}": count
                    for (stage, mode, severity), count in sorted(by_stage_mode.items())
                },
            }
        )
    return rows


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 90}")
    print("Taxa de falha por condição (ADR-003)")
    print(f"{'=' * 90}")
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_strategy[row["strategy"]].append(row)

    for strategy in sorted(by_strategy):
        print(f"\n{strategy}:")
        for row in sorted(by_strategy[strategy], key=lambda r: (r["model"], r["lang"])):
            n_conf, conf_rate = row["n_inter_agent_conflicts"], row["inter_agent_conflict_rate"]
            conflict_str = (
                f" | conflicts={n_conf} ({conf_rate:.2%})" if conf_rate is not None else ""
            )
            fail_str = f"failure_rate={row['failure_rate']:.2%}"
            fail_str += f" ({row['n_fatal_items']}/{row['n_dataset']})"
            flag_str = f"flagged_rate={row['flagged_rate']:.2%}"
            flag_str += f" ({row['n_flagged_items']}/{row['n_dataset']})"
            print(f"  {row['model']:<16} {row['lang']}  {fail_str}  {flag_str}{conflict_str}")

    # Agregado por modo, cross-arquitetura — responde "o pipeline alucina mais que o baseline?"
    print(f"\n{'=' * 90}")
    print("Agregado por modo de falha × arquitetura (soma de registros, todas condições)")
    print(f"{'=' * 90}")
    mode_by_strategy: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for key, count in row["by_stage_mode_severity"].items():
            _stage, mode, _severity = key.split("/")
            mode_by_strategy[row["strategy"]][mode] += count
    all_modes = sorted({m for c in mode_by_strategy.values() for m in c})
    header = f"{'mode':<24}" + "".join(f"{s:<20}" for s in sorted(mode_by_strategy))
    print(header)
    for mode in all_modes:
        line = f"{mode:<24}"
        for strategy in sorted(mode_by_strategy):
            line += f"{mode_by_strategy[strategy].get(mode, 0):<20}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/results/failure_rates.json")
    )
    args = parser.parse_args()

    rows = compute_failure_rates(args.results_dir)
    if not rows:
        raise SystemExit(f"Nenhuma condição com manifest.json+results.json em {args.results_dir}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_summary(rows)
    print(f"\n{len(rows)} condições agregadas -> {args.output}")


if __name__ == "__main__":
    main()
