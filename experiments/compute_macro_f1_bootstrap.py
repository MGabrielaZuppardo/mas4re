"""Paired, class-stratified bootstrap of the Macro-F1 differences (paper section 6).

The Wilcoxon and McNemar tests compare per-requirement correctness. This script
tests the Macro-F1 difference directly: for every (model, language) it resamples
requirements with replacement, separately within each true class (F, NF), and
recomputes the Macro-F1 of both architectures on the same resample.

Contrasts (each on the requirements valid, i.e. not a confidence-zero fallback,
for both sides): pipeline - baseline, two-call - baseline (Delta_p) and
pipeline - two-call (Delta_s).

Usage (from the repository root):
    python experiments/compute_macro_f1_bootstrap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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

BOOTSTRAP_RESAMPLES = 10_000
SEED = 42
N_COMPARISONS = 6
OUTPUT_PATH = RESULTS_DIR / "macro_f1_bootstrap.json"
CONTRASTS = {
    "pipeline_minus_baseline": ("pipeline", "baseline"),
    "delta_p_two_call_minus_baseline": ("two_call_baseline", "baseline"),
    "delta_s_pipeline_minus_two_call": ("pipeline", "two_call_baseline"),
}


def _is_valid(prediction: dict) -> bool:
    label = prediction.get("metadata", {}).get("label_type", "")
    return (
        prediction.get("confidence") != 0.0
        and bool(label)
        and bool(prediction.get("requirement_type"))
    )


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for cls in (0, 1):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def _encode(prediction: dict) -> tuple[int, int]:
    truth = int(prediction["metadata"]["label_type"].upper() != "F")
    pred = int(prediction["requirement_type"].upper() != "F")
    return truth, pred


def bootstrap_contrast(runs_a: dict, runs_b: dict, rng: np.random.Generator) -> dict | None:
    texts = sorted(
        t for t in set(runs_a) & set(runs_b) if _is_valid(runs_a[t]) and _is_valid(runs_b[t])
    )
    if len(texts) < 30:
        return None
    truth = np.array([_encode(runs_a[t])[0] for t in texts])
    pred_a = np.array([_encode(runs_a[t])[1] for t in texts])
    pred_b = np.array([_encode(runs_b[t])[1] for t in texts])
    observed = _macro_f1(truth, pred_a) - _macro_f1(truth, pred_b)

    strata = [np.where(truth == cls)[0] for cls in (0, 1)]
    deltas = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        deltas[i] = _macro_f1(truth[idx], pred_a[idx]) - _macro_f1(truth[idx], pred_b[idx])

    low, high = np.percentile(deltas, [2.5, 97.5])
    tail = 50 * 0.05 / N_COMPARISONS
    adj_low, adj_high = np.percentile(deltas, [tail, 100 - tail])
    p_two_sided = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return {
        "n": len(texts),
        "delta_macro_f1": round(observed, 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "ci_bonferroni": [round(float(adj_low), 4), round(float(adj_high), 4)],
        "p_bootstrap": round(min(float(p_two_sided), 1.0), 4),
        "excludes_zero_bonferroni": bool(adj_low > 0 or adj_high < 0),
    }


def main() -> None:
    all_runs = load_all_runs(load_csv(_find_latest_csv()))
    all_runs.update(load_two_call_runs())
    rng = np.random.default_rng(SEED)
    results = []
    for model in MODELS:
        for lang in LANGS:
            row: dict = {"model": model, "lang": lang}
            for name, (arch_a, arch_b) in CONTRASTS.items():
                a = all_runs.get((arch_a, model, lang))
                b = all_runs.get((arch_b, model, lang))
                row[name] = bootstrap_contrast(a, b, rng) if a and b else None
            results.append(row)
    payload = {
        "method": "paired bootstrap, stratified by true class, B=10000, seed=42",
        "n_comparisons_bonferroni": N_COMPARISONS,
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in results:
        cells = []
        for name in CONTRASTS:
            c = row[name]
            cells.append(
                f"{name.split('_')[0]}: {c['delta_macro_f1']:+.3f} {c['ci95']} n={c['n']}"
                if c
                else f"{name}: -"
            )
        print(f"{row['model']:<12} {row['lang']}  " + " | ".join(cells))
    print(f"\nwritten: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
