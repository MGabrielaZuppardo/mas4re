"""
MAS4RE — Reanálise estatística com McNemar (teste pareado para binário)

RQ1: Pipeline vs Baseline — McNemar exato (mesmo pareamento do compute_stats.py)
Efeito de idioma: PT vs EN — McNemar exato, pareado pelo campo `text`
(invariante em PT entre as duas condições), em vez do Mann-Whitney U
não-pareado usado em compute_stats.py.

Uso (a partir de D:/mas4re):
    python experiments/compute_mcnemar_stats.py                  # usa o CSV mais recente
    python experiments/compute_mcnemar_stats.py <caminho_csv>    # CSV específico

McNemar exato: teste binomial de duas caudas sobre o menor dos dois pares
discordantes (b = A correto/B incorreto, c = A incorreto/B correto),
n = b + c, H0: p = 0.5.
Bonferroni: RQ1 6 comparações → α′ = 0.05/6 = 0.0083
            Idioma 9 comparações → α′ = 0.05/9 = 0.0056
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.compute_stats import (
    ALPHA,
    LANGS,
    MODELS,
    RESULTS_DIR,
    _find_latest_csv,
    correct_binary,
    load_all_runs,
    load_csv,
    load_two_call_runs,
)

OUTPUT_PATH = RESULTS_DIR / "statistical_results_mcnemar.json"


def mcnemar_exact(pairs: list[tuple[int, int]]) -> tuple[int, int, float]:
    """McNemar exato sobre uma lista de pares binários (a, b).

    b_disc = casos onde a=1, b=0 (só a acerta); c_disc = casos onde a=0, b=1
    (só b acerta). Pares concordantes (a=b) são descartados, como no McNemar
    clássico. Retorna (b_disc, c_disc, p_value).
    """
    b_disc = sum(1 for a, b in pairs if a == 1 and b == 0)
    c_disc = sum(1 for a, b in pairs if a == 0 and b == 1)
    n = b_disc + c_disc
    if n == 0:
        return b_disc, c_disc, 1.0
    result = binomtest(min(b_disc, c_disc), n, 0.5, alternative="two-sided")
    return b_disc, c_disc, result.pvalue


def run_rq1_mcnemar(all_runs: dict) -> list[dict]:
    n_comparisons = 6
    alpha_bonf = ALPHA / n_comparisons
    print(f"\n{'=' * 72}")
    print("RQ1 — Pipeline vs Baseline  (McNemar exato, reanálise)")
    print(f"Bonferroni α′ = {ALPHA}/{n_comparisons} = {alpha_bonf:.4f}")
    print("=" * 72)

    results = []
    for model in MODELS:
        for lang in LANGS:
            base_run = all_runs.get(("baseline", model, lang), {})
            pipe_run = all_runs.get(("pipeline", model, lang), {})
            common = sorted(set(base_run) & set(pipe_run))
            pairs = [
                (correct_binary(pipe_run[t]), correct_binary(base_run[t]))
                for t in common
                if correct_binary(base_run[t]) is not None
                and correct_binary(pipe_run[t]) is not None
            ]
            if not pairs:
                print(f"  {model:<14} {lang.upper()}  SKIP (sem dados)")
                continue

            b_disc, c_disc, p_value = mcnemar_exact(pairs)
            sig_bonf = "sig" if p_value < alpha_bonf else "NS(Bonf)"
            results.append(
                {
                    "model": model,
                    "lang": lang,
                    "n": len(pairs),
                    "pipeline_only_correct": b_disc,
                    "baseline_only_correct": c_disc,
                    "p_value": p_value,
                    "p_bonferroni": min(p_value * n_comparisons, 1.0),
                    "sig_bonferroni": sig_bonf,
                }
            )
            print(
                f"  {model:<14} {lang.upper()}  n={len(pairs)}  "
                f"discordant(pipe-only={b_disc}, base-only={c_disc})  "
                f"p={p_value:.4g}  Bonf:{sig_bonf}"
            )
    return results


def run_language_mcnemar(all_runs: dict) -> list[dict]:
    n_comparisons = 9
    alpha_bonf = ALPHA / n_comparisons
    print(f"\n{'=' * 72}")
    print("Efeito de idioma — PT vs EN  (McNemar exato pareado, reanálise)")
    print("Pareado por `text` (invariante em PT entre condições EN e PT)")
    print(f"Bonferroni α′ = {ALPHA}/{n_comparisons} = {alpha_bonf:.4f}")
    print("=" * 72)

    results = []
    for strategy in ["baseline", "pipeline", "two_call_baseline"]:
        for model in MODELS:
            pt_run = all_runs.get((strategy, model, "pt"), {})
            en_run = all_runs.get((strategy, model, "en"), {})
            common = sorted(set(pt_run) & set(en_run))
            pairs = [
                (correct_binary(pt_run[t]), correct_binary(en_run[t]))
                for t in common
                if correct_binary(pt_run[t]) is not None and correct_binary(en_run[t]) is not None
            ]
            if not pairs:
                print(f"  {strategy:<18} {model:<14}  SKIP (sem pares alinhados)")
                continue

            b_disc, c_disc, p_value = mcnemar_exact(pairs)
            sig_bonf = "sig" if p_value < alpha_bonf else "NS(Bonf)"
            results.append(
                {
                    "strategy": strategy,
                    "model": model,
                    "n_paired": len(pairs),
                    "pt_only_correct": b_disc,
                    "en_only_correct": c_disc,
                    "p_value": p_value,
                    "p_bonferroni": min(p_value * n_comparisons, 1.0),
                    "sig_bonferroni": sig_bonf,
                }
            )
            print(
                f"  {strategy:<18} {model:<14}  n_paired={len(pairs)}  "
                f"discordant(pt-only={b_disc}, en-only={c_disc})  "
                f"p={p_value:.4g}  Bonf:{sig_bonf}"
            )
    return results


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _find_latest_csv()
    print(f"CSV: {csv_path}")

    run_meta = load_csv(csv_path)
    all_runs = load_all_runs(run_meta)
    all_runs.update(load_two_call_runs())
    print(f"Condições carregadas: {len(all_runs)}/18")

    rq1 = run_rq1_mcnemar(all_runs)
    language = run_language_mcnemar(all_runs)

    output = {
        "rq1_pipeline_vs_baseline_mcnemar": rq1,
        "language_effect_mcnemar_paired": language,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
