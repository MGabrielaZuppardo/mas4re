"""
MAS4RE — Contagem do reparo "tipo NF" por condição, a partir dos logs de execução

Alguns modelos (em especial o mistral:7b) devolvem o código da categoria NFR (ex.: "PE",
"SE") no campo requirement_type. Os agentes reinterpretam esse valor como NF e usam o
código como categoria, registrando um aviso "Schema fix". O reparo não deixa marca em
results.json, então a contagem vem dos logs filtrados em experiments/logs/.

Considera apenas as execuções completas (n=625) e conta requisições distintas por
condição. Saída: experiments/results/nf_repair_counts.json.

Uso (a partir da raiz do repositório):
    python experiments/compute_nf_repair_counts.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path("experiments/logs")
OUTPUT_PATH = Path("experiments/results/nf_repair_counts.json")
LOGS = {
    "grid_full.filtered.log": None,
    "ablation_grid.filtered.log": "two_call_baseline",
}
CONDITION_LINE = re.compile(r"INICIANDO \| (?:(\w+) \| )?([\w.:]+) \| (pt|en) \| n=(\d+)")
REPAIR_LINE = re.compile(r"\| (agents\.\w+) \| Schema fix: .*req_id=(\S+)")
FULL_RUN_SIZE = 625


def parse_log(path: Path, default_strategy: str | None) -> list[dict]:
    conditions: list[dict] = []
    current: dict | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        match = CONDITION_LINE.search(line)
        if match:
            strategy, model, lang, n = match.groups()
            current = {
                "strategy": strategy or default_strategy,
                "model": model,
                "lang": lang,
                "n": int(n),
                "source": path.name,
                "modules": defaultdict(set),
            }
            conditions.append(current)
            continue
        repair = REPAIR_LINE.search(line)
        if repair and current is not None:
            module, req_id = repair.groups()
            current["modules"][module].add(req_id)
    return conditions


def main() -> None:
    rows = []
    for name, default_strategy in LOGS.items():
        for condition in parse_log(LOG_DIR / name, default_strategy):
            if condition["n"] != FULL_RUN_SIZE:
                continue
            repaired = (
                set().union(*condition["modules"].values()) if condition["modules"] else set()
            )
            rows.append(
                {
                    "source": condition["source"],
                    "strategy": condition["strategy"],
                    "model": condition["model"],
                    "lang": condition["lang"],
                    "n": condition["n"],
                    "repairs": len(repaired),
                    "rate": round(len(repaired) / condition["n"], 4),
                    "modules": sorted(condition["modules"]),
                }
            )

    print(f"{'strategy':<18}{'model':<14}{'L':<4}{'repairs':>8}{'rate':>8}  module")
    for row in rows:
        print(
            f"{row['strategy']:<18}{row['model']:<14}{row['lang'].upper():<4}"
            f"{row['repairs']:>8}{row['rate']:>8.4f}  {','.join(row['modules']) or '-'}"
        )

    OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSalvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
