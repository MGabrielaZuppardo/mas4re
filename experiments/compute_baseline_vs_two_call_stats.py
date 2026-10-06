"""
MAS4RE — Ad-hoc validation script: baseline vs two_call_baseline.

This is the ONE comparison the paper never directly reports: it holds
the typed-state contract constant at "none" in both arms and varies
only decomposition (baseline = 1 call, no decomposition; two_call =
2 calls, decomposed, still no typed contract). RQ1 (baseline vs
pipeline) confounds decomposition+typing together; RQ2 (two_call vs
pipeline) isolates typing alone. This script isolates decomposition
alone, using the same protocol (paired Wilcoxon on binary correctness,
Cohen's h, Bonferroni alpha'=0.0083) as compute_stats.py / RQ1.

Usage (from D:/mas4re):
    python experiments/compute_baseline_vs_two_call_stats.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path("experiments/results")
ABLATION_DIR = RESULTS_DIR / "ablation_20260601T232947"

MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
ALPHA = 0.05
N_COMPARISONS = 6
ALPHA_BONF = ALPHA / N_COMPARISONS


def cohen_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def effect_label(h: float) -> str:
    a = abs(h)
    if a < 0.20:
        return "negligible"
    if a < 0.50:
        return "small"
    if a < 0.80:
        return "medium"
    return "large"


def _load_predictions(run_dir: Path) -> list[dict]:
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    return data["predictions"]


def _binary_outcomes(preds: list[dict]) -> dict[str, int]:
    # Paired by requirement text, not id: ids are generated per run and do
    # not match across runs collected at different times (verified: baseline
    # and two_call_baseline share 0 ids but 623/625 texts). This mirrors
    # compute_stats.py's own pairing key for RQ1.
    return {
        p["text"]: int(
            p.get("requirement_type", "") == p.get("metadata", {}).get("ground_truth_type", "")
        )
        for p in preds
    }


def _find_run_dir(results_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(results_dir.glob(f"{prefix}_*"))
    return candidates[-1] if candidates else None


def main() -> None:
    print("=" * 72)
    print("MAS4RE — baseline vs two_call_baseline (pure decomposition effect)")
    print(f"Bonferroni alpha' = {ALPHA}/{N_COMPARISONS} = {ALPHA_BONF:.4f}")
    print("=" * 72)

    for model in MODELS:
        slug = model.replace(":", "-")
        for lang in LANGS:
            base_dir = _find_run_dir(RESULTS_DIR, f"baseline_ollama-{slug}_{lang}_n625")
            two_dir = _find_run_dir(ABLATION_DIR, f"two_call_baseline_ollama-{slug}_{lang}_n625")

            if base_dir is None or two_dir is None:
                print(f"  [{model} {lang}] MISSING run dir — skip")
                continue

            base_out = _binary_outcomes(_load_predictions(base_dir))
            two_out = _binary_outcomes(_load_predictions(two_dir))

            common = sorted(set(base_out) & set(two_out))
            base_vec = np.array([base_out[i] for i in common])
            two_vec = np.array([two_out[i] for i in common])

            if not np.all(base_vec == two_vec):
                _, p_val = stats.wilcoxon(two_vec, base_vec, alternative="greater")
            else:
                p_val = 1.0

            p_base = base_vec.mean()
            p_two = two_vec.mean()
            h_val = cohen_h(p_two, p_base)
            sig = "yes" if p_val < ALPHA_BONF else "no"

            print(
                f"  {model:<14} {lang.upper()}  n={len(common):<4} "
                f"baseline={p_base:.4f}  two_call={p_two:.4f}  "
                f"delta={p_two - p_base:+.4f}  h={h_val:+.3f} ({effect_label(h_val)})  "
                f"p={p_val:.4g}  sig={sig}"
            )


if __name__ == "__main__":
    main()
