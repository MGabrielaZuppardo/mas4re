"""
MAS4RE — Ablation statistical analysis

Compares TwoCallBaseline vs Pipeline using the same RQ1 protocol:
    Wilcoxon signed-rank + Cohen's h + Bonferroni α' = 0.05/6 = 0.0083

Usage (from D:/mas4re):
    python experiments/compute_ablation_stats.py \\
        --ablation experiments/results/ablation_<timestamp>/ablation_summary.csv \\
        --grid    experiments/results/grid_summary_<timestamp>.csv

Output:
    Console table + ablation_stats_<timestamp>.json in the ablation directory

Interpretation guide (printed at end):
    ΔF1(pipeline − two_call_baseline) > 0  and  significant
        → typed state contract contributes beyond prompt specialisation
    ΔF1 ≈ 0  or  not significant
        → gain comes from specialised prompts alone
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats

# ── Statistical helpers ────────────────────────────────────────────────────────


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
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


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_predictions(run_dir: Path) -> list[dict]:
    """Load per-requirement predictions from a run directory's results.json."""
    results_file = run_dir / "results.json"
    if not results_file.exists():
        return []
    data = json.loads(results_file.read_text(encoding="utf-8"))
    return data.get("predictions", [])


def _load_ground_truth(run_dir: Path) -> dict[str, str]:
    """Build {req_id: true_type} from predictions (ground_truth_type field)."""
    preds = _load_predictions(run_dir)
    return {p["id"]: p.get("ground_truth_type", "") for p in preds}


def _binary_outcomes(preds: list[dict]) -> dict[str, int]:
    """Return {text: 1 if correct else 0} for binary FR/NFR classification.

    Keyed by requirement text, not by "id": Requirement.id is a random UUID
    regenerated on every PromiseAdapter.load() call, so it never aligns
    across two separate script invocations (e.g. pipeline run vs two-call
    run). Text is the stable join key across runs (same convention as
    compute_stats.py). Ground truth lives at metadata.label_type, not at a
    top-level "ground_truth_type" key.
    """
    return {
        p["text"]: int(
            p.get("requirement_type", "") == p.get("metadata", {}).get("label_type", "MISMATCH")
        )
        for p in preds
    }


def _find_run_dir(results_dir: Path, strategy: str, model_slug: str, lang: str) -> Path | None:
    """Locate the run directory matching (strategy, model, lang)."""
    pattern = f"{strategy}_{model_slug}_{lang}_*"
    candidates = sorted(results_dir.glob(pattern))
    return candidates[-1] if candidates else None


# ── Main analysis ──────────────────────────────────────────────────────────────

MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
ALPHA = 0.05
N_COMPARISONS = 6  # 3 models × 2 languages
ALPHA_BONF = ALPHA / N_COMPARISONS


def main() -> None:
    parser = argparse.ArgumentParser(description="MAS4RE Ablation Statistics")
    parser.add_argument(
        "--ablation",
        required=True,
        help="Path to ablation_summary.csv produced by run_ablation_grid.py",
    )
    parser.add_argument(
        "--grid",
        default=None,
        help="Path to grid_summary*.csv from the main experiment (optional: "
        "used to print ΔF1 vs pipeline side-by-side). If omitted, only "
        "two_call_baseline metrics are reported.",
    )
    args = parser.parse_args()

    ablation_csv = Path(args.ablation)
    ablation_dir = ablation_csv.parent

    # ── Load summary CSVs ─────────────────────────────────────────────────────
    def _read_csv(path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    ablation_rows = {
        (r["model"].split("/")[-1], r["lang"]): r
        for r in _read_csv(ablation_csv)
        if r["status"] == "ok"
    }

    pipeline_rows: dict[tuple[str, str], dict] = {}
    if args.grid:
        for r in _read_csv(Path(args.grid)):
            if r.get("strategy") == "pipeline" and r["status"] == "ok":
                pipeline_rows[(r["model"].split("/")[-1], r["lang"])] = r

    # ── Per-condition comparison ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("MAS4RE — Ablation: TwoCallBaseline vs Pipeline")
    print(f"Bonferroni α' = {ALPHA}/{N_COMPARISONS} = {ALPHA_BONF:.4f}")
    print("=" * 72)

    comparisons = []

    for model in MODELS:
        for lang in LANGS:
            key = (model, lang)
            abl = ablation_rows.get(key)
            pip = pipeline_rows.get(key)

            if abl is None:
                print(f"  [{model} {lang}] MISSING in ablation CSV — skip")
                continue

            f1_abl = float(abl["f1_macro"]) if abl["f1_macro"] else None
            f1_pip = float(pip["f1_macro"]) if (pip and pip.get("f1_macro")) else None
            delta = (f1_pip - f1_abl) if (f1_pip and f1_abl) else None

            # Paired Wilcoxon on per-requirement binary outcomes
            # Requires per-requirement prediction files in the run directories
            abl_run_dir = _find_run_dir(
                ablation_dir,
                "two_call_baseline",
                f"ollama-{model.replace(':', '-')}",
                lang,
            )
            pip_results_dir = ablation_dir.parent  # one level up from ablation dir
            pip_run_dir = _find_run_dir(
                pip_results_dir,
                "pipeline",
                f"ollama-{model.replace(':', '-')}-ollama-{model.replace(':', '-')}",
                lang,
            )

            w_stat, p_val, h_val, n_pairs = None, None, None, None

            if abl_run_dir and pip_run_dir:
                abl_preds = _load_predictions(abl_run_dir)
                pip_preds = _load_predictions(pip_run_dir)

                abl_outcomes = _binary_outcomes(abl_preds)
                pip_outcomes = _binary_outcomes(pip_preds)

                common_ids = sorted(set(abl_outcomes) & set(pip_outcomes))
                if len(common_ids) >= 10:
                    abl_vec = np.array([abl_outcomes[i] for i in common_ids])
                    pip_vec = np.array([pip_outcomes[i] for i in common_ids])
                    n_pairs = len(common_ids)

                    if not np.all(abl_vec == pip_vec):
                        w_stat, p_val = stats.wilcoxon(pip_vec, abl_vec, alternative="greater")
                    else:
                        p_val = 1.0

                    p_abl = abl_vec.mean()
                    p_pip = pip_vec.mean()
                    h_val = cohen_h(p_pip, p_abl)

            sig = "✓" if (p_val is not None and p_val < ALPHA_BONF) else "—"
            h_lab = effect_label(h_val) if h_val is not None else "n/a"

            row = {
                "model": model,
                "lang": lang,
                "f1_two_call": f1_abl,
                "f1_pipeline": f1_pip,
                "delta_f1": delta,
                "h": round(h_val, 3) if h_val is not None else None,
                "h_label": h_lab,
                "p": round(p_val, 5) if p_val is not None else None,
                "significant": sig == "✓",
                "n_pairs": n_pairs,
            }
            comparisons.append(row)

            # Console output
            f1_abl_s = f"{f1_abl:.4f}" if f1_abl else " n/a  "
            f1_pip_s = f"{f1_pip:.4f}" if f1_pip else " n/a  "
            delta_s = f"{delta:+.4f}" if delta else "  n/a "
            h_s = f"{h_val:+.3f}" if h_val is not None else "  n/a"
            p_s = f"{p_val:.4f}" if p_val is not None else "  n/a"
            print(
                f"  {model:<16} {lang.upper()}  "
                f"2call={f1_abl_s}  pipe={f1_pip_s}  "
                f"Δ={delta_s}  h={h_s} ({h_lab:<10})  p={p_s}  {sig}"
            )

    print()
    print("Legend: ✓ = significant after Bonferroni  |  Δ = F1(pipeline) − F1(two_call_baseline)")
    print()

    # ── Interpretation ─────────────────────────────────────────────────────────
    sig_count = sum(1 for r in comparisons if r["significant"])
    positive_delta = sum(1 for r in comparisons if r["delta_f1"] and r["delta_f1"] > 0)
    n_total = len(comparisons)

    print("─" * 72)
    print("INTERPRETATION")
    print(f"  Significant (Bonferroni-corrected): {sig_count}/{n_total} conditions")
    print(f"  Pipeline > TwoCallBaseline (ΔF1 > 0): {positive_delta}/{n_total} conditions")
    print()
    if sig_count >= 4:
        print("  → Typed inter-agent state contributes BEYOND prompt specialisation.")
        print("    The pipeline advantage is not solely due to the two task-specific prompts.")
    elif sig_count == 0:
        print("  → Advantage is attributable to PROMPT SPECIALISATION alone.")
        print("    The typed state contract (Pydantic validation + confidence routing)")
        print("    does not add measurable F1 benefit on top of the specialised prompts.")
    else:
        print("  → Mixed evidence. Consider model-capacity interaction.")
    print("─" * 72)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    from datetime import datetime

    out_path = ablation_dir / f"ablation_stats_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
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
