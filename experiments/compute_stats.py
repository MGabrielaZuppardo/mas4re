"""
MAS4RE — Cálculo estatístico completo (v2)

RQ1: Pipeline vs Baseline — Wilcoxon signed-rank + Cohen's h (pareado por requisito)
RQ2: Idioma EN vs PT     — Mann-Whitney U + Cohen's h (não-pareado)
OBS: MoSCoW consistency  — Fleiss' kappa descritivo (4 estratos, sem GT)

Uso (a partir de D:/mas4re):
    python experiments/compute_stats.py                  # usa o CSV mais recente
    python experiments/compute_stats.py <caminho_csv>    # CSV específico

Cohen's h: medida de effect size para proporções binárias (Cohen, 1988).
Thresholds: negligible < 0.20 | small 0.20–0.50 | medium 0.50–0.80 | large ≥ 0.80
Bonferroni: 6 comparações por RQ → α′ = 0.05/6 = 0.0083
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

# ── Configuração ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"
OUTPUT_PATH = RESULTS_DIR / "statistical_results.json"

MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
ALPHA = 0.05


# ── Localizar CSV de resumo ───────────────────────────────────────────────────


def _find_latest_csv() -> Path:
    """Retorna o CSV grid_summary mais recente gerado por run_grid_full.py."""
    candidates = sorted(RESULTS_DIR.glob("grid_summary_*.csv"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum grid_summary*.csv encontrado em {RESULTS_DIR}.\n"
            "Execute run_grid_full.py ou run_grid_rerun_fixes.py primeiro."
        )
    return candidates[0]


def load_csv(csv_path: Path) -> dict[str, dict]:
    """Carrega o CSV de resumo e retorna {run_id: meta_dict}."""
    meta: dict[str, dict] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["run_id"]] = {
                "strategy": row["strategy"],
                "model": row["model"].replace("ollama/", ""),
                "lang": row["lang"],
            }
    return meta


# ── Localizar results.json por run_id ────────────────────────────────────────


def find_results(run_id: str) -> Path | None:
    """
    M1-fix: tenta primeiro o path direto (run_id == nome do diretório após
    M1-fix). Fallback: busca por timestamp para compatibilidade com runs
    anteriores ao fix.
    """
    # Tentativa direta (pós M1-fix: run_id = nome exato do diretório)
    direct = RESULTS_DIR / run_id / "results.json"
    if direct.exists():
        return direct

    # Fallback: busca por timestamp (pré-M1-fix ou archive)
    ts = run_id.split("_")[-1]
    if ts.isdigit():
        for delta in range(-2, 3):
            ts_try = str(int(ts) + delta)
            candidates = list(RESULTS_DIR.glob(f"*{ts_try}/results.json"))
            if candidates:
                return candidates[0]
    return None


# ── Carregar predições ────────────────────────────────────────────────────────


def load_all_runs(run_meta: dict[str, dict]) -> dict[tuple, dict]:
    """Indexa predições por (strategy, model, lang) → {text: prediction_dict}."""
    all_runs: dict[tuple, dict] = {}
    missing = 0
    for run_id, meta in run_meta.items():
        path = find_results(run_id)
        if path is None or path.stat().st_size < 1000:
            print(f"  MISS: {run_id}")
            missing += 1
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        key = (meta["strategy"], meta["model"], meta["lang"])
        # Join key: Requirement.text é sempre PT (invariante ao C2-fix),
        # garantindo alinhamento consistente entre condições EN e PT.
        all_runs[key] = {p["text"]: p for p in data["predictions"]}
        print(f"  OK  {key}  n={len(all_runs[key])}")
    if missing:
        print(f"\n  ⚠  {missing} condição(ões) não localizadas — resultados parciais.")
    return all_runs


def load_two_call_runs() -> dict[tuple, dict]:
    """Indexa predições do two_call_baseline (ablation) por (strategy, model, lang) → {text: pred}.

    Vive em um diretório separado (experiments/results/ablation_<timestamp>/),
    fora do grid_summary*.csv principal, por isso não é coberto por load_all_runs().
    """
    ablation_dir = RESULTS_DIR / "ablation_20260601T232947"
    out: dict[tuple, dict] = {}
    for model in MODELS:
        slug = f"ollama-{model.replace(':', '-')}"
        for lang in LANGS:
            candidates = sorted(ablation_dir.glob(f"two_call_baseline_{slug}_{lang}_*"))
            if not candidates:
                print(f"  MISSING two_call {model} {lang}")
                continue
            results_file = candidates[-1] / "results.json"
            data = json.loads(results_file.read_text(encoding="utf-8"))
            key = ("two_call_baseline", model, lang)
            out[key] = {p["text"]: p for p in data["predictions"]}
            print(f"  OK  {key}  n={len(out[key])}")
    return out


# ── Funções estatísticas ──────────────────────────────────────────────────────


def correct_binary(pred: dict) -> int | None:
    gt = pred.get("metadata", {}).get("label_type", "").upper()
    rt = pred.get("requirement_type", "").upper()
    if not gt or not rt:
        return None
    return int(gt == rt)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h = 2·arcsin(√p1) − 2·arcsin(√p2)  (Cohen, 1988)."""

    def clamp(x):
        return max(0.0, min(1.0, x))

    return 2 * math.asin(math.sqrt(clamp(p1))) - 2 * math.asin(math.sqrt(clamp(p2)))


def h_magnitude(h: float) -> str:
    ah = abs(h)
    if ah < 0.20:
        return "negligible"
    if ah < 0.50:
        return "small"
    if ah < 0.80:
        return "medium"
    return "large"


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fleiss_kappa(matrix: list[list]) -> float:
    """Fleiss' κ — matrix shape: (n_subjects, n_raters), categorical values."""
    n = len(matrix)
    k_raters = len(matrix[0])
    cats = sorted({v for row in matrix for v in row})
    c = len(cats)
    cat_idx = {cat: i for i, cat in enumerate(cats)}
    P_mat = np.zeros((n, c))
    for i, row in enumerate(matrix):
        for v in row:
            P_mat[i, cat_idx[v]] += 1
    P_mat /= k_raters
    P_bar_i = (np.sum(P_mat**2, axis=1) - 1 / k_raters) / (1 - 1 / k_raters)
    P_bar = float(P_bar_i.mean())
    p_j = P_mat.mean(axis=0)
    P_e = float(np.sum(p_j**2))
    if abs(1 - P_e) < 1e-10:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def landis_koch(k: float) -> str:
    if k < 0:
        return "poor"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


# ── RQ1: Pipeline vs Baseline ─────────────────────────────────────────────────


def run_rq1(all_runs: dict) -> list[dict]:
    N = 6  # 3 modelos × 2 línguas
    alpha_bonf = ALPHA / N
    print(f"\n{'=' * 72}")
    print("RQ1 — Pipeline vs Baseline  (Wilcoxon pareado + Cohen h)")
    print(f"Bonferroni α′ = {ALPHA}/{N} = {alpha_bonf:.4f}")
    print("=" * 72)

    results = []
    for model in MODELS:
        for lang in LANGS:
            base_run = all_runs.get(("baseline", model, lang), {})
            pipe_run = all_runs.get(("pipeline", model, lang), {})
            common = sorted(set(base_run) & set(pipe_run))
            pairs = [
                (correct_binary(base_run[t]), correct_binary(pipe_run[t]))
                for t in common
                if correct_binary(base_run[t]) is not None
                and correct_binary(pipe_run[t]) is not None
            ]
            if not pairs:
                print(f"  {model:<14} {lang.upper()}  SKIP (sem dados)")
                continue

            bs = [p[0] for p in pairs]
            ps = [p[1] for p in pairs]
            n = len(pairs)
            mean_b = float(np.mean(bs))
            mean_p = float(np.mean(ps))
            h = cohens_h(mean_p, mean_b)
            mag = h_magnitude(h)
            try:
                _, pval = stats.wilcoxon(ps, bs, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                pval = 1.0
            sig_raw = sig_stars(pval)
            sig_bonf = "sig" if pval < alpha_bonf else "NS(Bonf)"

            results.append(
                {
                    "model": model,
                    "lang": lang,
                    "n": n,
                    "baseline_acc": round(mean_b, 6),
                    "pipeline_acc": round(mean_p, 6),
                    "delta_acc": round(mean_p - mean_b, 6),
                    "cohens_h": round(h, 6),
                    "magnitude": mag,
                    "p_value": pval,
                    "sig": sig_raw,
                    "p_bonferroni": min(pval * N, 1.0),
                    "sig_bonferroni": sig_bonf,
                }
            )
            print(
                f"  {model:<14} {lang.upper()}  n={n}  "
                f"base={mean_b:.4f}  pipe={mean_p:.4f}  "
                f"Δ={mean_p - mean_b:+.4f}  "
                f"h={h:+.4f}({mag})  "
                f"p={pval:.4e}{sig_raw}  Bonf:{sig_bonf}"
            )
    return results


# ── RQ2: Idioma EN vs PT ──────────────────────────────────────────────────────


def run_rq2_language(all_runs: dict) -> list[dict]:
    N = 9  # 3 modelos × 3 arquiteturas (baseline, pipeline, two_call_baseline)
    alpha_bonf = ALPHA / N
    print(f"\n{'=' * 72}")
    print("RQ2 — Idioma EN vs PT  (Mann-Whitney U não-pareado + Cohen h)")
    print("Nota: texto EN e PT são traduções distintas — comparação não-pareada")
    print(f"Bonferroni α′ = {ALPHA}/{N} = {alpha_bonf:.4f}")
    print("=" * 72)

    results = []
    for strategy in ["baseline", "pipeline", "two_call_baseline"]:
        for model in MODELS:
            pt_run = all_runs.get((strategy, model, "pt"), {})
            en_run = all_runs.get((strategy, model, "en"), {})
            pt_scores = [
                correct_binary(v) for v in pt_run.values() if correct_binary(v) is not None
            ]
            en_scores = [
                correct_binary(v) for v in en_run.values() if correct_binary(v) is not None
            ]
            if not pt_scores or not en_scores:
                print(f"  {strategy:<10} {model:<14}  SKIP (sem dados)")
                continue

            _, pval = stats.mannwhitneyu(pt_scores, en_scores, alternative="two-sided")
            mean_pt = float(np.mean(pt_scores))
            mean_en = float(np.mean(en_scores))
            h = cohens_h(mean_pt, mean_en)
            mag = h_magnitude(h)
            sig = sig_stars(pval)
            sig_bonf = "sig" if pval < alpha_bonf else "NS(Bonf)"

            results.append(
                {
                    "strategy": strategy,
                    "model": model,
                    "pt_acc": round(mean_pt, 6),
                    "en_acc": round(mean_en, 6),
                    "delta_pt_minus_en": round(mean_pt - mean_en, 6),
                    "cohens_h": round(h, 6),
                    "magnitude": mag,
                    "p_value": pval,
                    "sig": sig,
                    "p_bonferroni": min(pval * N, 1.0),
                    "sig_bonferroni": sig_bonf,
                }
            )
            print(
                f"  {strategy:<10} {model:<14}  "
                f"PT={mean_pt:.4f}  EN={mean_en:.4f}  "
                f"Δ(PT-EN)={mean_pt - mean_en:+.4f}  "
                f"h={h:+.4f}({mag})  "
                f"p={pval:.4e}{sig}  Bonf:{sig_bonf}"
            )
    return results


# ── Observação MoSCoW: Fleiss' kappa ─────────────────────────────────────────


def run_moscow_obs(all_runs: dict) -> list[dict]:
    print(f"\n{'=' * 72}")
    print("MoSCoW (observação arquitetural) — Fleiss' κ descritivo, sem GT")
    print("4 estratos: 2 arquiteturas × 2 línguas, 3 modelos como raters")
    print("=" * 72)

    results = []
    for strategy in ["baseline", "pipeline"]:
        for lang in LANGS:
            runs_by_model = {m: all_runs.get((strategy, m, lang), {}) for m in MODELS}
            common_texts = set(runs_by_model[MODELS[0]])
            for m in MODELS[1:]:
                common_texts &= set(runs_by_model[m])
            n_aligned = len(common_texts)

            if n_aligned < 10:
                print(f"\n  {strategy.upper():<10} × {lang.upper()}  SKIP (n_aligned={n_aligned})")
                continue

            VALID_PRIORITIES = {"M", "S", "C", "W"}
            valid_texts = {
                t
                for t in common_texts
                if all(runs_by_model[m][t].get("priority", "C") in VALID_PRIORITIES for m in MODELS)
            }
            n_excluded = n_aligned - len(valid_texts)
            if n_excluded:
                print(
                    f"  → {n_excluded} req(s) excluído(s) por priority "
                    "realmente malformado (fora de M/S/C/W)"
                )
            matrix = [
                [runs_by_model[m][t].get("priority", "C") for m in MODELS]
                for t in sorted(valid_texts)
            ]
            kappa = fleiss_kappa(matrix)
            interp = landis_koch(kappa)
            results.append(
                {
                    "strategy": strategy,
                    "lang": lang,
                    "n_aligned": len(valid_texts),
                    "n_excluded_invalid_priority": n_excluded,
                    "kappa": round(kappa, 6),
                    "interpretation": interp,
                    "distributions": {
                        m: dict(
                            Counter(runs_by_model[m][t].get("priority", "C") for t in valid_texts)
                        )
                        for m in MODELS
                    },
                }
            )
            print(f"\n  Stratum: {strategy.upper():<10} × {lang.upper()}  (n={len(valid_texts)})")
            print(f"  Fleiss κ = {kappa:.4f}  [{interp}]")
            for m in MODELS:
                vals = [runs_by_model[m][t].get("priority", "C") for t in common_texts]
                cnt = Counter(vals)
                total = sum(cnt.values())
                print(
                    f"    {m:<14}: "
                    f"M={cnt.get('M', 0) / total:.2f}  "
                    f"S={cnt.get('S', 0) / total:.2f}  "
                    f"C={cnt.get('C', 0) / total:.2f}  "
                    f"W={cnt.get('W', 0) / total:.2f}"
                )
    return results


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _find_latest_csv()
    print(f"CSV: {csv_path}")

    run_meta = load_csv(csv_path)
    print(f"Condições no CSV: {len(run_meta)}")

    print("\nCarregando predições...")
    all_runs = load_all_runs(run_meta)
    all_runs.update(load_two_call_runs())
    print(f"Condições carregadas: {len(all_runs)}/18")

    rq1 = run_rq1(all_runs)
    rq2 = run_rq2_language(all_runs)
    moscow = run_moscow_obs(all_runs)

    output = {
        "rq1_pipeline_vs_baseline": rq1,
        "rq2_language_en_vs_pt": rq2,
        "moscow_observation": moscow,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
