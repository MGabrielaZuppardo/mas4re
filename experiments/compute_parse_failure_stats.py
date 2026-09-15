"""
MAS4RE — Reanálise retendo falhas de parsing (confidence == 0.0)

O artigo exclui os 38 outputs com confidence == 0.0 (falhas de parsing do
extrator JSON compartilhado) da análise primária de F1/acurácia/MCC e dos
testes pareados (§6, política de exclusão uniforme). Este script recalcula
RQ1 (Wilcoxon) e o efeito de idioma (Mann-Whitney U) retendo todos os 625
requisitos e contando as falhas de parsing como predições incorretas, para
avaliar a sensibilidade das conclusões a essa política.

Uso (a partir de D:/mas4re):
    python experiments/compute_parse_failure_stats.py
    python experiments/compute_parse_failure_stats.py <caminho_csv>

Bonferroni: RQ1 6 comparações → α′ = 0.05/6 = 0.0083
            Idioma 9 comparações → α′ = 0.05/9 = 0.0056
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

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

OUTPUT_PATH = RESULTS_DIR / "statistical_results_retained.json"


def cohens_h(p1: float, p2: float) -> float:
    def clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    return 2 * math.asin(math.sqrt(clamp(p1))) - 2 * math.asin(math.sqrt(clamp(p2)))


def metrics_excluded_vs_retained(preds: dict) -> dict | None:
    """Acc/F1/MCC computados excluindo vs. retendo confidence == 0.0.

    Retorna None se a condição não tem nenhuma falha de parsing (idêntica
    nas duas políticas, não vale a pena reportar).
    """
    y_true_all, y_pred_all = [], []
    y_true_excl, y_pred_excl = [], []
    n_total = 0
    n_excluded = 0
    for p in preds.values():
        gt = p.get("metadata", {}).get("label_type", "").upper()
        rt = p.get("requirement_type", "").upper()
        conf = p.get("confidence")
        if not gt or not rt:
            continue
        n_total += 1
        y_true_all.append(gt)
        y_pred_all.append(rt)
        if conf == 0.0:
            n_excluded += 1
            continue
        y_true_excl.append(gt)
        y_pred_excl.append(rt)

    if n_excluded == 0:
        return None

    def scores(y_true: list, y_pred: list) -> tuple[float, float, float]:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except ValueError:
            mcc = 0.0
        return acc, f1, mcc

    acc_x, f1_x, mcc_x = scores(y_true_excl, y_pred_excl)
    acc_r, f1_r, mcc_r = scores(y_true_all, y_pred_all)
    return {
        "n_total": n_total,
        "n_excluded": n_excluded,
        "excluded": {"acc": acc_x, "f1_macro": f1_x, "mcc": mcc_x},
        "retained": {"acc": acc_r, "f1_macro": f1_r, "mcc": mcc_r},
        "delta_acc": acc_r - acc_x,
        "delta_f1_macro": f1_r - f1_x,
    }


def run_rq1_retained(all_runs: dict) -> list[dict]:
    n_comparisons = 6
    alpha_bonf = ALPHA / n_comparisons
    print(f"\n{'=' * 72}")
    print("RQ1 — Pipeline vs Baseline  (Wilcoxon, retendo falhas de parsing)")
    print(f"Bonferroni α′ = {ALPHA}/{n_comparisons} = {alpha_bonf:.4f}")
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
                continue

            bs = [p[0] for p in pairs]
            ps = [p[1] for p in pairs]
            stat, p_value = stats.wilcoxon(ps, bs, alternative="two-sided", zero_method="wilcox")
            h = cohens_h(float(np.mean(ps)), float(np.mean(bs)))
            sig_bonf = "sig" if p_value < alpha_bonf else "NS(Bonf)"
            results.append(
                {
                    "model": model,
                    "lang": lang,
                    "n": len(pairs),
                    "baseline_acc": round(float(np.mean(bs)), 6),
                    "pipeline_acc": round(float(np.mean(ps)), 6),
                    "cohens_h": round(h, 6),
                    "p_value": float(p_value),
                    "p_bonferroni": min(float(p_value) * n_comparisons, 1.0),
                    "sig_bonferroni": sig_bonf,
                }
            )
            print(
                f"  {model:<14} {lang.upper()}  n={len(pairs)}  "
                f"base={np.mean(bs):.4f}  pipe={np.mean(ps):.4f}  "
                f"h={h:+.4f}  p={p_value:.4g}  Bonf:{sig_bonf}"
            )
    return results


def run_language_retained(all_runs: dict) -> list[dict]:
    n_comparisons = 9
    alpha_bonf = ALPHA / n_comparisons
    print(f"\n{'=' * 72}")
    print("Efeito de idioma — PT vs EN  (Mann-Whitney U, retendo falhas de parsing)")
    print(f"Bonferroni α′ = {ALPHA}/{n_comparisons} = {alpha_bonf:.4f}")
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
                continue

            stat, p_value = stats.mannwhitneyu(pt_scores, en_scores, alternative="two-sided")
            h = cohens_h(float(np.mean(pt_scores)), float(np.mean(en_scores)))
            sig_bonf = "sig" if p_value < alpha_bonf else "NS(Bonf)"
            results.append(
                {
                    "strategy": strategy,
                    "model": model,
                    "pt_acc": round(float(np.mean(pt_scores)), 6),
                    "en_acc": round(float(np.mean(en_scores)), 6),
                    "cohens_h": round(h, 6),
                    "p_value": float(p_value),
                    "p_bonferroni": min(float(p_value) * n_comparisons, 1.0),
                    "sig_bonferroni": sig_bonf,
                }
            )
            print(
                f"  {strategy:<18} {model:<14}  "
                f"PT={np.mean(pt_scores):.4f}  EN={np.mean(en_scores):.4f}  "
                f"h={h:+.4f}  p={p_value:.4g}  Bonf:{sig_bonf}"
            )
    return results


def _acc(run: dict, exclude_zero_conf: bool) -> float:
    vals = []
    for p in run.values():
        gt = p.get("metadata", {}).get("label_type", "").upper()
        rt = (p.get("requirement_type") or "").upper()
        conf = p.get("confidence")
        if not gt or not rt:
            continue
        if exclude_zero_conf and conf == 0.0:
            continue
        vals.append(int(gt == rt))
    return float(np.mean(vals)) if vals else 0.0


def run_rq2_retained(all_runs: dict) -> list[dict]:
    """Delta_p (two-call - baseline) e Delta_s (pipeline - two-call), Tabela 9,
    excluindo vs. retendo falhas de parsing. Isola a politica de exclusao como
    unica variavel, usando a mesma logica de alinhamento em ambos os casos."""
    print(f"\n{'=' * 72}")
    print("RQ2 (Tabela 9) — Delta_p / Delta_s, excluído vs. retido")
    print("=" * 72)

    results = []
    for model in MODELS:
        for lang in LANGS:
            base = all_runs.get(("baseline", model, lang), {})
            two = all_runs.get(("two_call_baseline", model, lang), {})
            pipe = all_runs.get(("pipeline", model, lang), {})
            if not base or not two or not pipe:
                continue

            ba_x, tw_x, pi_x = _acc(base, True), _acc(two, True), _acc(pipe, True)
            ba_r, tw_r, pi_r = _acc(base, False), _acc(two, False), _acc(pipe, False)
            dp_x, ds_x = tw_x - ba_x, pi_x - tw_x
            dp_r, ds_r = tw_r - ba_r, pi_r - tw_r
            changed = abs(dp_x - dp_r) > 0.0005 or abs(ds_x - ds_r) > 0.0005

            results.append(
                {
                    "model": model,
                    "lang": lang,
                    "delta_p_excluded": round(dp_x, 6),
                    "delta_p_retained": round(dp_r, 6),
                    "delta_s_excluded": round(ds_x, 6),
                    "delta_s_retained": round(ds_r, 6),
                    "changed": changed,
                }
            )
            print(
                f"  {model:<14} {lang.upper()}  "
                f"Δp: excl={dp_x:+.4f} ret={dp_r:+.4f}   "
                f"Δs: excl={ds_x:+.4f} ret={ds_r:+.4f}"
                f"{'  <- CHANGED' if changed else ''}"
            )
    return results


def run_metric_deltas(all_runs: dict) -> list[dict]:
    print(f"\n{'=' * 72}")
    print("Acc/F1/MCC — excluído (artigo) vs. retido, condições com falhas de parsing")
    print("=" * 72)

    results = []
    for strategy in ["baseline", "pipeline", "two_call_baseline"]:
        for model in MODELS:
            for lang in LANGS:
                preds = all_runs.get((strategy, model, lang), {})
                if not preds:
                    continue
                delta = metrics_excluded_vs_retained(preds)
                if delta is None:
                    continue
                results.append({"strategy": strategy, "model": model, "lang": lang, **delta})
                print(
                    f"  {strategy:<18} {model:<14} {lang.upper()}  "
                    f"n_excluded={delta['n_excluded']}/{delta['n_total']}  "
                    f"Δacc={delta['delta_acc']:+.4f}  Δf1={delta['delta_f1_macro']:+.4f}"
                )
    return results


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _find_latest_csv()
    print(f"CSV: {csv_path}")

    run_meta = load_csv(csv_path)
    all_runs = load_all_runs(run_meta)
    all_runs.update(load_two_call_runs())
    print(f"Condições carregadas: {len(all_runs)}/18")

    rq1 = run_rq1_retained(all_runs)
    rq2 = run_rq2_retained(all_runs)
    language = run_language_retained(all_runs)
    metric_deltas = run_metric_deltas(all_runs)

    output = {
        "rq1_pipeline_vs_baseline_retained": rq1,
        "rq2_delta_p_delta_s_retained": rq2,
        "language_effect_retained": language,
        "metric_deltas_excluded_vs_retained": metric_deltas,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
