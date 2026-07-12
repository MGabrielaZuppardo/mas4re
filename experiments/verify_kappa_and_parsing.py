"""
Ad-hoc verification script (not part of the paper pipeline):

1. Leave-one-model-out Fleiss' kappa for the 6 strata in tab:kappa
   (baseline/pipeline from grid_summary, two_call_baseline from ablation dir).
2. Invalid-priority ("parsing failure") rate by model x lang x strategy.

Run from D:/mas4re:
    python experiments/verify_kappa_and_parsing.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compute_stats import fleiss_kappa, landis_koch, load_all_runs, load_csv

RESULTS_DIR = Path("experiments/results")
MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
VALID_PRIORITIES = {"M", "S", "C", "W"}


def load_two_call_runs() -> dict[tuple, dict]:
    """Load two_call_baseline predictions keyed by (strategy, model, lang) -> {text: pred}."""
    ablation_dir = RESULTS_DIR / "ablation_20260601T232947"
    out = {}
    for model in MODELS:
        slug = f"ollama-{model.replace(':', '-')}"
        for lang in LANGS:
            candidates = sorted(ablation_dir.glob(f"two_call_baseline_{slug}_{lang}_*"))
            if not candidates:
                print(f"  MISSING two_call {model} {lang}")
                continue
            results_file = candidates[-1] / "results.json"
            data = json.loads(results_file.read_text(encoding="utf-8"))
            out[("two_call_baseline", model, lang)] = {p["text"]: p for p in data["predictions"]}
    return out


def main() -> None:
    csv_path = Path("experiments/results/grid_summary_nfull_20260524T184454.csv")
    run_meta = load_csv(csv_path)
    all_runs = load_all_runs(run_meta)
    all_runs.update(load_two_call_runs())

    print("\n" + "=" * 78)
    print("1) LEAVE-ONE-MODEL-OUT FLEISS' KAPPA (per stratum)")
    print("=" * 78)

    for strategy in ["baseline", "pipeline", "two_call_baseline"]:
        for lang in LANGS:
            runs_by_model = {m: all_runs.get((strategy, m, lang), {}) for m in MODELS}
            if any(not r for r in runs_by_model.values()):
                print(f"  {strategy:<18} {lang.upper()}  SKIP (missing data)")
                continue

            common_texts = set(runs_by_model[MODELS[0]])
            for m in MODELS[1:]:
                common_texts &= set(runs_by_model[m])

            valid_texts_3 = {
                t
                for t in common_texts
                if all(runs_by_model[m][t].get("priority", "C") in VALID_PRIORITIES for m in MODELS)
            }
            matrix_3 = [
                [runs_by_model[m][t].get("priority", "C") for m in MODELS]
                for t in sorted(valid_texts_3)
            ]
            k_full = fleiss_kappa(matrix_3)

            print(f"\n  Stratum: {strategy:<18} {lang.upper()}  (n={len(valid_texts_3)}, 3 raters)")
            print(f"    Full 3-rater kappa = {k_full:+.4f} [{landis_koch(k_full)}]")

            for dropped in MODELS:
                kept = [m for m in MODELS if m != dropped]
                valid_texts_2 = {
                    t
                    for t in common_texts
                    if all(
                        runs_by_model[m][t].get("priority", "C") in VALID_PRIORITIES for m in kept
                    )
                }
                matrix_2 = [
                    [runs_by_model[m][t].get("priority", "C") for m in kept]
                    for t in sorted(valid_texts_2)
                ]
                k_2 = fleiss_kappa(matrix_2)
                delta = k_2 - k_full
                print(
                    f"    drop {dropped:<14} -> kappa({kept[0]},{kept[1]}) = {k_2:+.4f} "
                    f"[{landis_koch(k_2)}]  Δ={delta:+.4f}"
                )

    print("\n" + "=" * 78)
    print("2) INVALID-PRIORITY / PARSE-FAILURE RATE BY MODEL x LANG x STRATEGY")
    print("=" * 78)

    for strategy in ["baseline", "pipeline", "two_call_baseline"]:
        for lang in LANGS:
            for model in MODELS:
                run = all_runs.get((strategy, model, lang), {})
                if not run:
                    continue
                total = len(run)
                priorities = [p.get("priority", "C") for p in run.values()]
                cnt = Counter(priorities)
                invalid = sum(v for k, v in cnt.items() if k not in VALID_PRIORITIES)
                rate = invalid / total if total else 0.0
                flag = "  <-- " if rate > 0 else ""
                print(
                    f"  {strategy:<18} {lang.upper()}  {model:<14}  "
                    f"invalid={invalid:>3}/{total}  rate={rate:.4f}  dist={dict(cnt)}{flag}"
                )


if __name__ == "__main__":
    main()
