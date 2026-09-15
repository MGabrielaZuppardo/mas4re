"""
MAS4RE — Cohen's g: effect size pareado por contagem de pares discordantes

Cohen's h (Table 4, Table 9) usa apenas as proporções marginais de acerto
de cada arquitetura, ignorando o pareamento por requisito que o próprio
teste de Wilcoxon usa. Este script calcula Cohen's g (Cohen, 1988), o
effect size correto para uma proporção pareada testada contra 0.5 — a
mesma lógica usada pelo McNemar exato — sobre os pares discordantes
(onde as duas arquiteturas divergem).

g = P - 0.5, onde P = b / (b + c); b = só a primeira arquitetura acerta,
c = só a segunda acerta. Pares concordantes são descartados (não carregam
informação sobre qual arquitetura é melhor).

Thresholds (Cohen, 1988): |g| < 0.05 negligible | < 0.15 small |
                           < 0.25 medium | >= 0.25 large

Uso (a partir de D:/mas4re):
    python experiments/compute_cohens_g.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.compute_stats import (
    LANGS,
    MODELS,
    RESULTS_DIR,
    _find_latest_csv,
    correct_binary,
    load_all_runs,
    load_csv,
    load_two_call_runs,
)

OUTPUT_PATH = RESULTS_DIR / "cohens_g_paired.json"


def discordant_counts(a_run: dict, b_run: dict) -> tuple[int, int, int]:
    """Retorna (b_disc, c_disc, n_pares_totais) para os pares alinhados por texto.

    b_disc = só `a_run` acerta; c_disc = só `b_run` acerta.
    """
    common = sorted(set(a_run) & set(b_run))
    pairs = [
        (correct_binary(a_run[t]), correct_binary(b_run[t]))
        for t in common
        if correct_binary(a_run[t]) is not None and correct_binary(b_run[t]) is not None
    ]
    b_disc = sum(1 for a, b in pairs if a == 1 and b == 0)
    c_disc = sum(1 for a, b in pairs if a == 0 and b == 1)
    return b_disc, c_disc, len(pairs)


def cohens_g(b_disc: int, c_disc: int) -> float:
    n_disc = b_disc + c_disc
    if n_disc == 0:
        return 0.0
    return b_disc / n_disc - 0.5


def g_magnitude(g: float) -> str:
    ag = abs(g)
    if ag < 0.05:
        return "negligible"
    if ag < 0.15:
        return "small"
    if ag < 0.25:
        return "medium"
    return "large"


def main() -> None:
    csv_path = _find_latest_csv()
    run_meta = load_csv(csv_path)
    all_runs = load_all_runs(run_meta)
    all_runs.update(load_two_call_runs())

    print(f"\n{'=' * 88}")
    print("Table 4 (RQ1) — Cohen's g pareado (pipeline vs baseline)")
    print("=" * 88)
    rq1 = []
    for model in MODELS:
        for lang in LANGS:
            base_run = all_runs[("baseline", model, lang)]
            pipe_run = all_runs[("pipeline", model, lang)]
            b_disc, c_disc, n_pairs = discordant_counts(pipe_run, base_run)
            n_disc = b_disc + c_disc
            g = cohens_g(b_disc, c_disc)
            rq1.append(
                {
                    "model": model,
                    "lang": lang,
                    "n_pairs": n_pairs,
                    "n_discordant": n_disc,
                    "pipe_only_correct": b_disc,
                    "base_only_correct": c_disc,
                    "cohens_g": round(g, 4),
                    "magnitude": g_magnitude(g),
                }
            )
            print(
                f"  {model:<14} {lang.upper()}  n_disc={n_disc:>3}  "
                f"(pipe={b_disc}, base={c_disc})  g={g:+.4f}  [{g_magnitude(g)}]"
            )

    print(f"\n{'=' * 88}")
    print("Table 9 (Delta_state) — Cohen's g pareado (pipeline vs two-call)")
    print("Aviso: n_discordant muito baixo em varias condicoes -> estimativa instavel")
    print("=" * 88)
    table9 = []
    for model in MODELS:
        for lang in LANGS:
            two_run = all_runs[("two_call_baseline", model, lang)]
            pipe_run = all_runs[("pipeline", model, lang)]
            b_disc, c_disc, n_pairs = discordant_counts(pipe_run, two_run)
            n_disc = b_disc + c_disc
            g = cohens_g(b_disc, c_disc)
            table9.append(
                {
                    "model": model,
                    "lang": lang,
                    "n_pairs": n_pairs,
                    "n_discordant": n_disc,
                    "pipe_only_correct": b_disc,
                    "two_call_only_correct": c_disc,
                    "cohens_g": round(g, 4),
                    "magnitude": g_magnitude(g),
                    "low_power_warning": n_disc < 20,
                }
            )
            print(
                f"  {model:<14} {lang.upper()}  n_disc={n_disc:>3}  "
                f"(pipe={b_disc}, 2call={c_disc})  g={g:+.4f}  [{g_magnitude(g)}]"
                f"{'  <- n_disc<20, instavel' if n_disc < 20 else ''}"
            )

    output = {"rq1_table4_cohens_g": rq1, "delta_state_table9_cohens_g": table9}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
