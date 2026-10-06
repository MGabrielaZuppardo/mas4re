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

Além do ponto estimado, calcula o IC 95% exato (Clopper-Pearson) sobre a
proporção de pares discordantes e o converte para IC de g. Um IC estreito
o bastante para caber inteiro numa única faixa de magnitude sustenta essa
leitura com confiança; um IC que atravessa múltiplas faixas (ex.: de
"negligible" até "large") significa que o n de pares discordantes não é
suficiente para afirmar a magnitude do efeito com confiança — nesse caso
o veredito correto é "não estimável com confiança neste desenho", não
"negligible" (que é uma leitura tão arbitrária quanto qualquer outra
dentro do IC).

Uso (a partir de D:/mas4re):
    python experiments/compute_cohens_g.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import binomtest

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


def cohens_g_ci(b_disc: int, c_disc: int, confidence: float = 0.95) -> tuple[float, float]:
    """IC exato (Clopper-Pearson) de g = P - 0.5, via binomtest.proportion_ci
    sobre P = b_disc / (b_disc + c_disc)."""
    n_disc = b_disc + c_disc
    if n_disc == 0:
        return (0.0, 0.0)
    result = binomtest(b_disc, n_disc, 0.5, alternative="two-sided")
    ci = result.proportion_ci(confidence_level=confidence, method="exact")
    return (ci.low - 0.5, ci.high - 0.5)


NEGLIGIBLE_BAND = 0.05


def estimability_verdict(ci_low: float, ci_high: float) -> str:
    """O que importa para a alegação do artigo ("Δstate é negligible") não é
    se o IC cabe numa única sub-faixa fina de Cohen, mas se ele permite ou
    não excluir a hipótese de efeito negligible (|g| < 0.05).

    - Se o IC fica inteiro fora de [-0.05, +0.05] (mesmo sinal dos dois
      lados, |g|>=0.05 nos dois extremos): "negligible" está descartado com
      confiança — o efeito é real, magnitude aproximada dada pelo ponto.
    - Se o IC fica inteiro dentro de [-0.05, +0.05]: "negligible" confirmado
      com confiança (raro, exige n_discordant grande).
    - Caso contrário (o IC cruza a fronteira de 0.05 de qualquer lado):
      não dá para confirmar nem descartar "negligible" com confiança neste
      n de pares discordantes — é a zona onde afirmar "negligible" é uma
      leitura arbitrária dentre as compatíveis com o IC.
    """
    if ci_high < -NEGLIGIBLE_BAND or ci_low > NEGLIGIBLE_BAND:
        return "efeito_nao_negligivel_confirmado"
    if -NEGLIGIBLE_BAND <= ci_low and ci_high <= NEGLIGIBLE_BAND:
        return "negligible_confirmado"
    return "nao_estimavel_com_confianca"


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
            ci_low, ci_high = cohens_g_ci(b_disc, c_disc)
            verdict = estimability_verdict(ci_low, ci_high)
            rq1.append(
                {
                    "model": model,
                    "lang": lang,
                    "n_pairs": n_pairs,
                    "n_discordant": n_disc,
                    "pipe_only_correct": b_disc,
                    "base_only_correct": c_disc,
                    "cohens_g": round(g, 4),
                    "cohens_g_ci95": [round(ci_low, 4), round(ci_high, 4)],
                    "magnitude": g_magnitude(g),
                    "verdict": verdict,
                }
            )
            print(
                f"  {model:<14} {lang.upper()}  n_disc={n_disc:>3}  "
                f"(pipe={b_disc}, base={c_disc})  g={g:+.4f} [{ci_low:+.3f},{ci_high:+.3f}]  "
                f"[{g_magnitude(g)}]  {verdict}"
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
            ci_low, ci_high = cohens_g_ci(b_disc, c_disc)
            verdict = estimability_verdict(ci_low, ci_high)
            table9.append(
                {
                    "model": model,
                    "lang": lang,
                    "n_pairs": n_pairs,
                    "n_discordant": n_disc,
                    "pipe_only_correct": b_disc,
                    "two_call_only_correct": c_disc,
                    "cohens_g": round(g, 4),
                    "cohens_g_ci95": [round(ci_low, 4), round(ci_high, 4)],
                    "magnitude": g_magnitude(g),
                    "verdict": verdict,
                }
            )
            print(
                f"  {model:<14} {lang.upper()}  n_disc={n_disc:>3}  "
                f"(pipe={b_disc}, 2call={c_disc})  g={g:+.4f} [{ci_low:+.3f},{ci_high:+.3f}]  "
                f"[{g_magnitude(g)}]  {verdict}"
            )

    output = {"rq1_table4_cohens_g": rq1, "delta_state_table9_cohens_g": table9}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
