"""
MAS4RE — Tabela de consistência MoSCoW: Fleiss' kappa por (arquitetura x idioma)

Gera, a partir das predições do grid principal (baseline, pipeline) e da ablação
(two_call_baseline), o kappa de Fleiss entre os três modelos (tratados como três
avaliadores) em cada um dos 6 estratos, com os requisitos alinhados por texto, e a
taxa do rótulo Must (M) por modelo e arquitetura.

Substitui o antigo moscow_kappa_comparison.csv, que usava um critério de exclusão
anterior (n entre 616 e 623) e não tinha script gerador no repositório.

Uso (a partir da raiz do repositório):
    python experiments/compute_kappa_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.compute_stats import (
    RESULTS_DIR,
    _find_latest_csv,
    fleiss_kappa,
    load_all_runs,
    load_csv,
    load_two_call_runs,
)

MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
ARCHITECTURES = ["baseline", "two_call_baseline", "pipeline"]
VALID_PRIORITIES = {"M", "S", "C", "W"}
OUTPUT_PATH = RESULTS_DIR / "kappa_table.json"

LANDIS_KOCH = [
    (0.0, "poor"),
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.01, "almost perfect"),
]


def interpret(kappa: float) -> str:
    if kappa < 0.0:
        return "poor"
    return next(label for upper, label in LANDIS_KOCH[1:] if kappa <= upper)


def _priorities(runs: dict, architecture: str, model: str, lang: str) -> dict[str, str]:
    return {t: p.get("priority", "C") for t, p in runs[(architecture, model, lang)].items()}


def stratum(runs: dict, architecture: str, lang: str) -> dict:
    by_model = {m: _priorities(runs, architecture, m, lang) for m in MODELS}
    texts = sorted(set.intersection(*(set(by_model[m]) for m in MODELS)))
    matrix = [[by_model[m][t] for m in MODELS] for t in texts]
    out_of_set = sum(label not in VALID_PRIORITIES for row in matrix for label in row)
    kappa = float(fleiss_kappa(matrix))
    must_rate = {
        m: round(sum(by_model[m][t] == "M" for t in texts) / len(texts), 4) for m in MODELS
    }
    return {
        "n_aligned": len(texts),
        "out_of_set_labels": out_of_set,
        "kappa": round(kappa, 4),
        "interpretation": interpret(kappa),
        "must_rate": must_rate,
    }


def main() -> None:
    runs = load_all_runs(load_csv(_find_latest_csv()))
    runs.update(load_two_call_runs())

    table = {
        f"{architecture}_{lang}": stratum(runs, architecture, lang)
        for architecture in ARCHITECTURES
        for lang in LANGS
    }

    print(f"\n{'estrato':22s} {'n':>4s} {'kappa':>8s}  {'interp.':10s}  Must (qwen/llama/mistral)")
    for name, row in table.items():
        rates = "/".join(f"{row['must_rate'][m]:.2f}" for m in MODELS)
        print(
            f"{name:22s} {row['n_aligned']:4d} {row['kappa']:+8.4f}  "
            f"{row['interpretation']:10s}  {rates}"
        )

    OUTPUT_PATH.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
