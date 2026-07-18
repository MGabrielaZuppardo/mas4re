"""
MAS4RE — Bootstrap confidence interval for the ablation's delta F1-macro
(pipeline − two_call_baseline), complementing compute_ablation_stats.py.

compute_ablation_stats.py already reports Wilcoxon p-value and Cohen's h
on paired per-requirement binary correctness. This script adds a
percentile bootstrap 95% CI directly on delta F1-macro (the same metric
reported in the ablation table), to check whether "not significant" is
"no detectable effect" (CI tight around zero) rather than "underpowered"
(CI wide / one-sided).

Usage (from D:/mas4re):
    python experiments/compute_ablation_bootstrap.py

Output:
    Console table + ablation_bootstrap_<timestamp>.json in the ablation
    results directory.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import f1_score

from experiments.bootstrap_ci import paired_bootstrap_ci

RESULTS_DIR = Path("experiments/results")
ABLATION_DIR = RESULTS_DIR / "ablation_20260601T232947"

MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
N_BOOT = 10_000
SEED = 42
ALPHA = 0.05
N_COMPARISONS = 6  # same 6 model×lang comparisons as compute_ablation_stats.py
ALPHA_BONF = ALPHA / N_COMPARISONS


def _load_predictions(run_dir: Path) -> list[dict]:
    results_file = run_dir / "results.json"
    data = json.loads(results_file.read_text(encoding="utf-8"))
    return data["predictions"]


def _find_run_dir(base: Path, pattern: str) -> Path:
    candidates = sorted(base.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No run dir matching {pattern!r} under {base}")
    return candidates[-1]


def _paired_labels(
    two_call_preds: list[dict], pipeline_preds: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join two_call and pipeline predictions by requirement text (stable
    key across runs — Requirement.id is a random UUID regenerated on every
    PromiseAdapter.load(), same convention as compute_ablation_stats.py).

    Returns (y_true, y_pred_two_call, y_pred_pipeline) aligned arrays.
    """
    two_call_by_text = {p["text"]: p for p in two_call_preds}
    pipeline_by_text = {p["text"]: p for p in pipeline_preds}
    common = sorted(set(two_call_by_text) & set(pipeline_by_text))

    y_true, y_two_call, y_pipeline = [], [], []
    for text in common:
        tc = two_call_by_text[text]
        pl = pipeline_by_text[text]
        true_label = tc.get("metadata", {}).get("label_type", "")
        if not true_label:
            continue
        y_true.append(true_label)
        y_two_call.append(tc.get("requirement_type", "MISMATCH"))
        y_pipeline.append(pl.get("requirement_type", "MISMATCH"))

    return np.array(y_true), np.array(y_two_call), np.array(y_pipeline)


def bootstrap_delta_f1(
    y_true: np.ndarray,
    y_two_call: np.ndarray,
    y_pipeline: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    n = len(y_true)
    labels = sorted(set(y_true) | set(y_two_call) | set(y_pipeline))

    observed_f1_two_call = f1_score(y_true, y_two_call, labels=labels, average="macro")
    observed_f1_pipeline = f1_score(y_true, y_pipeline, labels=labels, average="macro")
    observed_delta = observed_f1_pipeline - observed_f1_two_call

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1_tc = f1_score(y_true[idx], y_two_call[idx], labels=labels, average="macro")
        f1_pl = f1_score(y_true[idx], y_pipeline[idx], labels=labels, average="macro")
        deltas[b] = f1_pl - f1_tc

    ci_result = paired_bootstrap_ci(observed_delta, deltas, ALPHA, N_COMPARISONS)

    return {
        "n_pairs": n,
        "f1_two_call": round(float(observed_f1_two_call), 4),
        "f1_pipeline": round(float(observed_f1_pipeline), 4),
        "delta_f1": round(float(observed_delta), 4),
        **ci_result,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    comparisons = []

    print("\n" + "=" * 88)
    print(f"MAS4RE — Ablation bootstrap 95% CI on ΔF1-macro (pipeline − two_call), B={N_BOOT}")
    print("=" * 88)

    for model in MODELS:
        model_slug = f"ollama-{model.replace(':', '-')}"
        for lang in LANGS:
            two_call_dir = _find_run_dir(ABLATION_DIR, f"two_call_baseline_{model_slug}_{lang}_*")
            pipeline_dir = _find_run_dir(
                RESULTS_DIR, f"pipeline_{model_slug}-{model_slug}_{lang}_*"
            )

            two_call_preds = _load_predictions(two_call_dir)
            pipeline_preds = _load_predictions(pipeline_dir)
            y_true, y_tc, y_pl = _paired_labels(two_call_preds, pipeline_preds)

            result = bootstrap_delta_f1(y_true, y_tc, y_pl, N_BOOT, rng)
            result["model"] = model
            result["lang"] = lang
            comparisons.append(result)

            print(
                f"  {model:<14} {lang.upper()}  n={result['n_pairs']:<4} "
                f"ΔF1={result['delta_f1']:+.4f}  "
                f"CI95=[{result['ci95_low']:+.4f}, {result['ci95_high']:+.4f}]  "
                f"{'incl.0' if result['ci_includes_zero'] else 'excl.0':<7}  "
                f"CI_Bonf=[{result['ci_bonferroni_low']:+.4f}, "
                f"{result['ci_bonferroni_high']:+.4f}]  "
                f"{'incl.0' if result['ci_bonferroni_includes_zero'] else 'excl.0':<7}  "
                f"p_boot={result['p_bootstrap_two_sided']:.4f}"
            )

    n_excludes_zero = sum(1 for r in comparisons if not r["ci_includes_zero"])
    n_bonf_excludes_zero = sum(1 for r in comparisons if not r["ci_bonferroni_includes_zero"])
    print()
    print(f"95% CI excludes zero in {n_excludes_zero}/{len(comparisons)} conditions.")
    print(
        f"Bonferroni-adjusted ({100 * (1 - ALPHA_BONF):.2f}%) CI excludes zero in "
        f"{n_bonf_excludes_zero}/{len(comparisons)} conditions."
    )
    print("=" * 88)

    out_path = ABLATION_DIR / f"ablation_bootstrap_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "n_boot": N_BOOT,
                "seed": SEED,
                "alpha_bonferroni": ALPHA_BONF,
                "n_comparisons": N_COMPARISONS,
                "comparisons": comparisons,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
