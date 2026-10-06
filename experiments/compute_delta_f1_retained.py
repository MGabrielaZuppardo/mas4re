"""
MAS4RE — Delta_p / Delta_s em F1 macro, excluindo vs. retendo falhas de parsing

A Tabela 9 do artigo reporta Delta_p e Delta_s em F1 macro, mas a analise de
sensibilidade em compute_parse_failure_stats.py (run_rq2_retained) usa acuracia.
Este script calcula as duas politicas sobre o mesmo F1 macro da Tabela 9, para
que o valor reportado para llama3.1:8b EN (cuja condicao baseline retem 33
fallbacks) seja exato e rastreavel, e nao uma subtracao de valores arredondados.

Excluir: descarta predicoes com confidence == 0.0 (fallback de parsing).
Reter:   conta essas predicoes como previsoes normais (FUNCTIONAL).

    Delta_p = F1(two_call) - F1(baseline)
    Delta_s = F1(pipeline) - F1(two_call)

Uso (a partir de D:/mas4re):
    python experiments/compute_delta_f1_retained.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.compute_stats import (
    LANGS,
    MODELS,
    RESULTS_DIR,
    _find_latest_csv,
    load_all_runs,
    load_csv,
    load_two_call_runs,
)

OUTPUT_PATH = RESULTS_DIR / "delta_f1_retained.json"


def macro_f1(run: dict, exclude_zero_conf: bool) -> float:
    y_true, y_pred = [], []
    for p in run.values():
        gt = p.get("metadata", {}).get("label_type", "").upper()
        rt = (p.get("requirement_type") or "").upper()
        if not gt or not rt:
            continue
        if exclude_zero_conf and p.get("confidence") == 0.0:
            continue
        y_true.append(gt)
        y_pred.append(rt)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def delta_row(model: str, lang: str, all_runs: dict) -> dict | None:
    base = all_runs.get(("baseline", model, lang), {})
    two = all_runs.get(("two_call_baseline", model, lang), {})
    pipe = all_runs.get(("pipeline", model, lang), {})
    if not (base and two and pipe):
        return None

    row: dict = {"model": model, "lang": lang}
    for policy, exclude in (("excluded", True), ("retained", False)):
        f1_base = macro_f1(base, exclude)
        f1_two = macro_f1(two, exclude)
        f1_pipe = macro_f1(pipe, exclude)
        row[policy] = {
            "f1_baseline": round(f1_base, 6),
            "f1_two_call": round(f1_two, 6),
            "f1_pipeline": round(f1_pipe, 6),
            "delta_p": round(f1_two - f1_base, 6),
            "delta_s": round(f1_pipe - f1_two, 6),
        }
    return row


def main() -> None:
    all_runs = load_all_runs(load_csv(_find_latest_csv()))
    all_runs.update(load_two_call_runs())
    rows = []
    for model in MODELS:
        for lang in LANGS:
            row = delta_row(model, lang, all_runs)
            if row is not None:
                rows.append(row)

    print(f"{'model':<14} {'L':<3} {'dp_excl':>8} {'dp_ret':>8} {'ds_excl':>8} {'ds_ret':>8}")
    for r in rows:
        x, t = r["excluded"], r["retained"]
        print(
            f"{r['model']:<14} {r['lang'].upper():<3} {x['delta_p']:>+8.4f} {t['delta_p']:>+8.4f} "
            f"{x['delta_s']:>+8.4f} {t['delta_s']:>+8.4f}"
        )

    OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSalvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
