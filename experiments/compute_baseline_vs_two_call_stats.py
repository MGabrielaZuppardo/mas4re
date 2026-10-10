"""
MAS4RE — Comparação direta baseline vs two_call_baseline (efeito de decomposição)

Mantém o contrato tipado fixado em "ausente" nos dois braços e varia apenas a
decomposição (baseline = 1 chamada; two_call = 2 chamadas com prompts específicos, sem
contrato tipado). É a verificação não subtrativa citada na seção de ablação do artigo.

Mesmo protocolo do RQ1 (compute_stats.py): correção binária pareada por texto do
requisito, Wilcoxon bilateral, Cohen's h, Bonferroni alpha' = 0.05/6. Reporta também o
McNemar exato e o Cohen's g pareado, e duas políticas para os fallbacks de parsing
(confidence == 0.0):

  excluded: descarta os pares em que qualquer dos dois braços caiu em fallback
            (política primária do artigo; n pode ser menor que 623);
  retained: mantém os fallbacks como previsões incorretas (n = 623).

Uso (a partir da raiz do repositório):
    python experiments/compute_baseline_vs_two_call_stats.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

RESULTS_DIR = Path("experiments/results")
ABLATION_DIR = RESULTS_DIR / "ablation_20260601T232947"
OUTPUT_PATH = RESULTS_DIR / "baseline_vs_two_call.json"

MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
ALPHA = 0.05
N_COMPARISONS = 6
ALPHA_BONF = ALPHA / N_COMPARISONS


def cohen_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def effect_label(h: float) -> str:
    magnitude = abs(h)
    if magnitude < 0.20:
        return "negligible"
    if magnitude < 0.50:
        return "small"
    if magnitude < 0.80:
        return "medium"
    return "large"


def load_predictions(run_dir: Path) -> dict[str, dict]:
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    return {p["text"]: p for p in data["predictions"]}


def _correct(prediction: dict) -> int:
    expected = prediction.get("metadata", {}).get("ground_truth_type", "")
    return int(prediction.get("requirement_type", "") == expected)


def _find_run_dir(results_dir: Path, prefix: str) -> Path:
    return sorted(results_dir.glob(f"{prefix}_*"))[-1]


def compare(baseline: dict[str, dict], two_call: dict[str, dict], exclude_fallbacks: bool) -> dict:
    texts = sorted(set(baseline) & set(two_call))
    if exclude_fallbacks:
        texts = [
            t
            for t in texts
            if baseline[t]["confidence"] != 0.0 and two_call[t]["confidence"] != 0.0
        ]
    base = np.array([_correct(baseline[t]) for t in texts])
    two = np.array([_correct(two_call[t]) for t in texts])

    two_only = int(np.sum((two == 1) & (base == 0)))
    base_only = int(np.sum((base == 1) & (two == 0)))
    discordant = two_only + base_only
    p_wilcoxon = (
        float(wilcoxon(two, base, alternative="two-sided", zero_method="wilcox").pvalue)
        if discordant
        else 1.0
    )
    p_mcnemar = float(binomtest(two_only, discordant, 0.5).pvalue) if discordant else 1.0
    h_value = cohen_h(float(two.mean()), float(base.mean()))
    return {
        "n": len(texts),
        "accuracy_baseline": round(float(base.mean()), 4),
        "accuracy_two_call": round(float(two.mean()), 4),
        "delta_accuracy": round(float(two.mean() - base.mean()), 4),
        "cohens_h": round(h_value, 4),
        "h_magnitude": effect_label(h_value),
        "two_call_only_correct": two_only,
        "baseline_only_correct": base_only,
        "cohens_g": round(two_only / discordant - 0.5, 4) if discordant else 0.0,
        "wilcoxon_p": p_wilcoxon,
        "mcnemar_exact_p": p_mcnemar,
        "significant_wilcoxon_bonferroni": bool(p_wilcoxon < ALPHA_BONF),
        "significant_mcnemar_bonferroni": bool(p_mcnemar < ALPHA_BONF),
    }


def main() -> None:
    report = []
    for model in MODELS:
        slug = model.replace(":", "-")
        for lang in LANGS:
            baseline = load_predictions(
                _find_run_dir(RESULTS_DIR, f"baseline_ollama-{slug}_{lang}_n625")
            )
            two_call = load_predictions(
                _find_run_dir(ABLATION_DIR, f"two_call_baseline_ollama-{slug}_{lang}_n625")
            )
            report.append(
                {
                    "model": model,
                    "lang": lang,
                    "excluded": compare(baseline, two_call, exclude_fallbacks=True),
                    "retained": compare(baseline, two_call, exclude_fallbacks=False),
                }
            )

    print(f"Bonferroni alpha' = {ALPHA}/{N_COMPARISONS} = {ALPHA_BONF:.4f}")
    for policy in ("excluded", "retained"):
        print(f"\n== {policy}")
        for row in report:
            r = row[policy]
            print(
                f"  {row['model']:<12} {row['lang'].upper()} n={r['n']:<4} "
                f"d_acc={r['delta_accuracy']:+.4f} h={r['cohens_h']:+.3f} g={r['cohens_g']:+.3f} "
                f"b/c={r['two_call_only_correct']}/{r['baseline_only_correct']} "
                f"p_W={r['wilcoxon_p']:.2e} p_M={r['mcnemar_exact_p']:.2e} "
                f"sig_M={'yes' if r['significant_mcnemar_bonferroni'] else 'no'}"
            )
        n_sig = sum(row[policy]["significant_mcnemar_bonferroni"] for row in report)
        print(f"  significativas (McNemar, Bonferroni): {n_sig} de {len(report)}")

    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSalvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
