"""
MAS4RE — Variância de não-determinismo por condição (temperature=0.0)

temperature=0.0 não garante decodificação determinística nos providers
LLM usados neste projeto (Ollama/Anthropic/Groq/Azure). Este script lê
um grid_summary.csv produzido por scripts/run_grid.py --repeat N>=2
(mesma condição, mesmo seed, executada N vezes) e reporta, por
(strategy, model, lang): média, desvio-padrão e amplitude de accuracy,
f1_macro e mcc entre as repetições.

Não executa nenhum experimento — apenas agrega repetições já coletadas.

Uso:
    python experiments/compute_determinism_variance.py \
        --summary experiments/results/grid_summary.csv \
        --output experiments/results/determinism_variance.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

METRICS = ("accuracy", "f1_macro", "mcc")


def _load_rows(summary_path: Path) -> list[dict]:
    with summary_path.open(encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "ok"]


def compute_variance(rows: list[dict]) -> list[dict]:
    """Agrupa por (strategy, model, lang) e calcula estatísticas entre repetições.

    Condições com apenas 1 repetição (repeat_idx="0" ou uma única linha)
    são incluídas com desvio-padrão None, para deixar explícito que a
    variância não foi medida — não são silenciosamente omitidas.
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["strategy"], row["model"], row["lang"])
        groups[key].append(row)

    results = []
    for (strategy, model, lang), group_rows in sorted(groups.items()):
        n_repeats = len(group_rows)
        entry: dict = {
            "strategy": strategy,
            "model": model,
            "lang": lang,
            "n_repeats": n_repeats,
        }
        for metric in METRICS:
            values = [float(r[metric]) for r in group_rows if r.get(metric)]
            if not values:
                continue
            entry[f"{metric}_mean"] = round(statistics.mean(values), 4)
            entry[f"{metric}_stdev"] = (
                round(statistics.stdev(values), 4) if len(values) > 1 else None
            )
            entry[f"{metric}_min"] = round(min(values), 4)
            entry[f"{metric}_max"] = round(max(values), 4)
            entry[f"{metric}_range"] = round(max(values) - min(values), 4)
        results.append(entry)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/results/grid_summary.csv"),
        help="CSV produzido por scripts/run_grid.py --repeat N",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/determinism_variance.json"),
    )
    args = parser.parse_args()

    if not args.summary.exists():
        raise SystemExit(
            f"Summary não encontrado: {args.summary}. "
            "Rode primeiro: python scripts/run_grid.py --repeat N"
        )

    rows = _load_rows(args.summary)
    if not rows:
        raise SystemExit(f"Nenhuma linha com status=ok em {args.summary}.")

    results = compute_variance(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(results)} condições agregadas -> {args.output}")
    for entry in results:
        if entry["n_repeats"] < 2:
            print(
                f"  {entry['strategy']:<18} {entry['model']:<25} {entry['lang']} "
                f"| n_repeats=1 (variância não medida)"
            )
            continue
        acc_stdev = entry.get("accuracy_stdev")
        print(
            f"  {entry['strategy']:<18} {entry['model']:<25} {entry['lang']} "
            f"| n_repeats={entry['n_repeats']} "
            f"| accuracy: mean={entry.get('accuracy_mean')} stdev={acc_stdev} "
            f"range={entry.get('accuracy_range')}"
        )


if __name__ == "__main__":
    main()
