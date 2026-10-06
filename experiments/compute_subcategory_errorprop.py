"""
MAS4RE — Análise complementar para §6
1. NFR subcategory metrics (10-class) por condição
2. Error propagation: impacto de confidence < 0.70 no pipeline
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Configuração ───────────────────────────────────────────────────────────────
results_dir = Path("D:/mas4re/experiments/results")
csv_path = "D:/mas4re/experiments/results/grid_summary_nfull_20260524T184454.csv"
CONF_THRESHOLD = 0.70

MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
NFR_CATS = ["A", "FT", "LF", "MN", "O", "PE", "PO", "SC", "SE", "US"]

# ── Carregar dados (reutiliza lógica do compute_stats.py) ──────────────────────
run_meta = {}
with open(csv_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        run_meta[row["run_id"]] = {
            "strategy": row["strategy"],
            "model": row["model"].replace("ollama/", ""),
            "lang": row["lang"],
        }


def find_results(run_id):
    ts = run_id.split("_")[-1]
    for delta in [0, -1, 1, -2, 2]:
        candidates = list(results_dir.glob(f"*{int(ts) + delta}/results.json"))
        if candidates:
            return candidates[0]
    return None


all_runs: dict[tuple, list] = {}  # key → lista de predictions (não dict por texto)
all_runs_by_text: dict[tuple, dict] = {}

for run_id, meta in run_meta.items():
    path = find_results(run_id)
    if not path or path.stat().st_size < 1000:
        continue
    with open(path, encoding="utf-8") as f:
        preds = json.load(f)["predictions"]
    key = (meta["strategy"], meta["model"], meta["lang"])
    all_runs[key] = preds
    all_runs_by_text[key] = {p["text"]: p for p in preds}

print(f"Condições carregadas: {len(all_runs)}/12\n")

# ══════════════════════════════════════════════════════════════════════════════
# 1. SUBCATEGORY METRICS — F1 por categoria NFR (10-class)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("1. NFR SUBCATEGORY METRICS (10-class, somente reqs ground-truth NF)")
print("=" * 72)

subcat_results = {}

for strategy in ["baseline", "pipeline"]:
    for model in MODELS:
        for lang in LANGS:
            key = (strategy, model, lang)
            preds = all_runs.get(key, [])
            if not preds:
                continue

            # y_true[i]/y_pred[i] span all 625 predictions in the condition.
            # y_true is 1 only for requirements that are ground-truth NF of
            # this category AND that the pipeline also classified as NF
            # (subcategory is only meaningful once the binary label is
            # right); y_pred is 1 whenever the model emitted this
            # subcategory label, regardless of its binary call. Passed
            # together to sklearn, this yields precision against every
            # emission of the label (a spurious subcategory on a
            # misclassified-as-FR item still counts as a false positive)
            # while recall/support stay restricted to the correctly-
            # classified-as-NF subset — matching Table tab:nfr exactly.
            gt_counts = Counter(
                p["metadata"].get("label_category", "UNKNOWN")
                for p in preds
                if p.get("metadata", {}).get("label_type", "").upper() == "NF"
                and p.get("requirement_type", "").upper() == "NF"
            )
            n_nf_correct = sum(gt_counts.values())

            f1_per_cat = {}
            p_per_cat = {}
            r_per_cat = {}
            for cat in NFR_CATS:
                support = gt_counts.get(cat, 0)
                if support == 0:
                    f1_per_cat[cat] = None
                    continue
                y_true = [
                    int(
                        p.get("metadata", {}).get("label_type", "").upper() == "NF"
                        and p.get("requirement_type", "").upper() == "NF"
                        and p["metadata"].get("label_category") == cat
                    )
                    for p in preds
                ]
                y_pred = [int(p.get("nfr_category") == cat) for p in preds]
                p_per_cat[cat] = round(precision_score(y_true, y_pred, zero_division=0), 4)
                r_per_cat[cat] = round(recall_score(y_true, y_pred, zero_division=0), 4)
                f1_per_cat[cat] = round(f1_score(y_true, y_pred, zero_division=0), 4)

            valid_cats = [c for c in NFR_CATS if gt_counts.get(c, 0) > 0]
            f1_macro_nfr = (
                float(np.mean([f1_per_cat[c] for c in valid_cats])) if valid_cats else 0.0
            )

            subcat_results[key] = {
                "f1_per_cat": f1_per_cat,
                "p_per_cat": p_per_cat,
                "r_per_cat": r_per_cat,
                "f1_macro_nfr": round(f1_macro_nfr, 4),
                "n_nf_correctly_classified": n_nf_correct,
                "gt_counts": dict(gt_counts),
            }

            label = f"{strategy:<10} {model:<14} {lang.upper()}"
            print(
                f"\n  {label}  |  NFR macro-F1={f1_macro_nfr:.4f}  "
                f"(n_NF_correctly_classified={n_nf_correct})"
            )
            print(f"  {'Cat':<5} {'n':>5}  {'F1':>6}  {'P':>6}  {'R':>6}")
            for cat in NFR_CATS:
                sup = gt_counts.get(cat, 0)
                f1v = f1_per_cat.get(cat)
                if f1v is None:
                    continue
                low = " << baixo suporte" if sup < 30 else ""
                print(
                    f"  {cat:<5} {sup:>5}  {f1v:>6.3f}  {p_per_cat[cat]:>6.3f}  "
                    f"{r_per_cat[cat]:>6.3f}{low}"
                )

# ══════════════════════════════════════════════════════════════════════════════
# 2. ERROR PROPAGATION — confidence < 0.70 no pipeline
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"2. ERROR PROPAGATION  (confidence < {CONF_THRESHOLD} no pipeline)")
print("=" * 72)

ep_results = {}

for model in MODELS:
    for lang in LANGS:
        pipe_key = ("pipeline", model, lang)
        base_key = ("baseline", model, lang)
        pipe_preds = all_runs_by_text.get(pipe_key, {})
        base_preds = all_runs_by_text.get(base_key, {})

        common = set(pipe_preds) & set(base_preds)

        # Separar em high-confidence e low-confidence pelo pipeline
        low_conf = []
        high_conf = []
        for text in common:
            pp = pipe_preds[text]
            bp = base_preds[text]
            conf = float(pp.get("confidence", 1.0))
            gt = pp.get("metadata", {}).get("label_type", "").upper()
            pipe_type = pp.get("requirement_type", "").upper()
            base_type = bp.get("requirement_type", "").upper()
            pipe_moscow = pp.get("priority", "C")
            base_moscow = bp.get("priority", "C")
            row = {
                "conf": conf,
                "gt": gt,
                "pipe_type": pipe_type,
                "base_type": base_type,
                "pipe_correct": int(gt == pipe_type),
                "base_correct": int(gt == base_type),
                "pipe_moscow": pipe_moscow,
                "base_moscow": base_moscow,
                "moscow_agree": int(pipe_moscow == base_moscow),
            }
            if conf < CONF_THRESHOLD:
                low_conf.append(row)
            else:
                high_conf.append(row)

        n_low = len(low_conf)
        n_high = len(high_conf)

        if n_low == 0:
            print(f"\n  {model} {lang.upper()}: nenhum req com confidence < {CONF_THRESHOLD}")
            continue

        # Accuracy nos dois subsets
        acc_pipe_low = np.mean([r["pipe_correct"] for r in low_conf])
        acc_pipe_high = (
            np.mean([r["pipe_correct"] for r in high_conf]) if high_conf else float("nan")
        )
        acc_base_low = np.mean([r["base_correct"] for r in low_conf])
        acc_base_high = (
            np.mean([r["base_correct"] for r in high_conf]) if high_conf else float("nan")
        )

        # Concordância MoSCoW
        moscow_agree_low = np.mean([r["moscow_agree"] for r in low_conf])
        moscow_agree_high = (
            np.mean([r["moscow_agree"] for r in high_conf]) if high_conf else float("nan")
        )

        # Distribuição de confiança nos low-conf
        confs_low = [r["conf"] for r in low_conf]

        ep_results[(model, lang)] = {
            "n_low": n_low,
            "n_high": n_high,
            "acc_pipe_low": acc_pipe_low,
            "acc_pipe_high": acc_pipe_high,
            "acc_base_low": acc_base_low,
            "acc_base_high": acc_base_high,
            "moscow_agree_low": moscow_agree_low,
            "moscow_agree_high": moscow_agree_high,
            "conf_mean_low": float(np.mean(confs_low)),
            "conf_min_low": float(np.min(confs_low)),
        }

        print(f"\n  {model} {lang.upper()}  |  low-conf n={n_low}  high-conf n={n_high}")
        print(
            f"  {'Subset':<12} {'n':>5}  {'pipe_acc':>9}  {'base_acc':>9}  {'Dacc':>7}  {'MoSCoW_agree':>13}"  # noqa: E501
        )
        print(
            f"  {'low(<0.70)':<12} {n_low:>5}  {acc_pipe_low:>9.4f}  {acc_base_low:>9.4f}  "
            f"{acc_pipe_low - acc_base_low:>+7.4f}  {moscow_agree_low:>13.4f}"
        )
        print(
            f"  {'high(>=0.70)':<12} {n_high:>5}  {acc_pipe_high:>9.4f}  {acc_base_high:>9.4f}  "
            f"{acc_pipe_high - acc_base_high:>+7.4f}  {moscow_agree_high:>13.4f}"
        )
        print(f"  Conf low: mean={np.mean(confs_low):.3f}  min={np.min(confs_low):.3f}")

# ── Salvar tudo ────────────────────────────────────────────────────────────────
out = {
    "subcategory": {str(k): v for k, v in subcat_results.items()},
    "error_propagation": {str(k): v for k, v in ep_results.items()},
}
out_path = Path("D:/mas4re/experiments/results/subcategory_errorprop.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nResultados salvos em: {out_path}")
