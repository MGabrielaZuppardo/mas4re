"""
MAS4RE — Bootstrap confidence interval for the RQ3 kappa increment
(pipeline - two_call_baseline) on Fleiss' kappa (cross-model MoSCoW
agreement), complementing compute_stats.py's descriptive RQ3 table.

compute_stats.py reports Fleiss' kappa per (strategy, lang) stratum
(Table 3) but does not test whether the pipeline > two_call_baseline
increment is distinguishable from sampling noise. This script adds a
paired percentile bootstrap 95% CI directly on delta kappa (pipeline
minus two_call_baseline): for each bootstrap resample, the same drawn
requirement indices are used to recompute kappa under both
architectures, so the two draws stay paired per resample -- mirroring
compute_ablation_bootstrap.py's approach for RQ2's delta F1-macro.

Usage (from D:/mas4re):
    python experiments/compute_rq3_kappa_bootstrap.py

Output:
    Console table + rq3_kappa_bootstrap_<timestamp>.json in
    experiments/results/.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from experiments.compute_stats import (
    _find_latest_ablation_csv,
    _find_latest_csv,
    fleiss_kappa,
    load_all_runs,
    load_csv,
)

MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
VALID_PRIORITIES = {"M", "S", "C"}
N_BOOT = 10_000
SEED = 42
ALPHA = 0.05
N_COMPARISONS = 2  # pt, en -- one delta-kappa comparison per language
ALPHA_BONF = ALPHA / N_COMPARISONS


def _priorities_by_model(all_runs: dict, strategy: str, lang: str) -> dict[str, dict[str, str]]:
    return {
        m: {t: p.get("priority", "C") for t, p in all_runs.get((strategy, m, lang), {}).items()}
        for m in MODELS
    }


def bootstrap_delta_kappa(
    common_texts: list[str],
    two_call_by_model: dict[str, dict[str, str]],
    pipeline_by_model: dict[str, dict[str, str]],
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    n = len(common_texts)
    matrix_tc = [[two_call_by_model[m][t] for m in MODELS] for t in common_texts]
    matrix_pl = [[pipeline_by_model[m][t] for m in MODELS] for t in common_texts]

    observed_kappa_tc = fleiss_kappa(matrix_tc)
    observed_kappa_pl = fleiss_kappa(matrix_pl)
    observed_delta = observed_kappa_pl - observed_kappa_tc

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_tc = [matrix_tc[i] for i in idx]
        boot_pl = [matrix_pl[i] for i in idx]
        deltas[b] = fleiss_kappa(boot_pl) - fleiss_kappa(boot_tc)

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    bonf_pct = [100 * ALPHA_BONF / 2, 100 * (1 - ALPHA_BONF / 2)]
    ci_bonf_low, ci_bonf_high = np.percentile(deltas, bonf_pct)
    # Fraction of resamples on the opposite side of 0 from the observed
    # delta (two-sided) -- a bootstrap-based analogue to a p-value.
    if observed_delta >= 0:
        p_boot = float(np.mean(deltas <= 0)) * 2
    else:
        p_boot = float(np.mean(deltas >= 0)) * 2
    p_boot = min(p_boot, 1.0)

    return {
        "n_aligned": n,
        "kappa_two_call": round(float(observed_kappa_tc), 4),
        "kappa_pipeline": round(float(observed_kappa_pl), 4),
        "delta_kappa": round(float(observed_delta), 4),
        "ci95_low": round(float(ci_low), 4),
        "ci95_high": round(float(ci_high), 4),
        "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
        "ci_bonferroni_low": round(float(ci_bonf_low), 4),
        "ci_bonferroni_high": round(float(ci_bonf_high), 4),
        "ci_bonferroni_includes_zero": bool(ci_bonf_low <= 0 <= ci_bonf_high),
        "p_bootstrap_two_sided": round(p_boot, 4),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)

    run_meta = load_csv(_find_latest_csv())
    ablation_csv = _find_latest_ablation_csv()
    if ablation_csv is None:
        raise FileNotFoundError(
            "No ablation_summary.csv found under experiments/results/*/ -- "
            "run the two_call_baseline grid first."
        )
    run_meta.update(load_csv(ablation_csv))

    print("Carregando predicoes...")
    all_runs = load_all_runs(run_meta)

    print("\n" + "=" * 88)
    print(f"MAS4RE - RQ3 bootstrap 95% CI on delta-kappa (pipeline - two_call), B={N_BOOT}")
    print("=" * 88)

    comparisons = []
    for lang in LANGS:
        tc_by_model = _priorities_by_model(all_runs, "two_call_baseline", lang)
        pl_by_model = _priorities_by_model(all_runs, "pipeline", lang)

        common = set(tc_by_model[MODELS[0]]) & set(pl_by_model[MODELS[0]])
        for m in MODELS[1:]:
            common &= set(tc_by_model[m]) & set(pl_by_model[m])
        common_texts = sorted(
            t
            for t in common
            if all(tc_by_model[m][t] in VALID_PRIORITIES for m in MODELS)
            and all(pl_by_model[m][t] in VALID_PRIORITIES for m in MODELS)
        )

        result = bootstrap_delta_kappa(common_texts, tc_by_model, pl_by_model, N_BOOT, rng)
        result["lang"] = lang
        comparisons.append(result)

        print(
            f"  {lang.upper():<3} n={result['n_aligned']:<4} "
            f"kappa(two_call)={result['kappa_two_call']:+.4f}  "
            f"kappa(pipeline)={result['kappa_pipeline']:+.4f}  "
            f"delta={result['delta_kappa']:+.4f}  "
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

    out_dir = Path("experiments/results")
    out_path = out_dir / f"rq3_kappa_bootstrap_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
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
