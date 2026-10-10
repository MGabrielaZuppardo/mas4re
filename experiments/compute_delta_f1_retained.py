"""
MAS4RE — Delta_p / Delta_s em F1 macro, excluindo vs. retendo falhas de parsing

A Tabela 9 do artigo reporta Delta_p e Delta_s em F1 macro sobre as 625 predicoes de
cada execucao, mas a analise de sensibilidade em compute_parse_failure_stats.py
(run_rq2_retained) usa acuracia sobre as 623 requisicoes alinhadas por texto. Este
script calcula as duas politicas sobre o mesmo F1 macro e a mesma base (625) da Tabela 9,
para que o valor reportado para llama3.1:8b EN (cuja condicao baseline retem 33
fallbacks) seja exato e rastreavel, e nao uma subtracao de valores arredondados.

Excluir: descarta predicoes com confidence == 0.0 (fallback de parsing).
Reter:   conta essas predicoes como previsoes normais (FUNCTIONAL).

    Delta_p = F1(two_call) - F1(baseline)
    Delta_s = F1(pipeline) - F1(two_call)

Uso (a partir da raiz do repositorio):
    python experiments/compute_delta_f1_retained.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from sklearn.metrics import f1_score

RESULTS_DIR = Path("experiments/results")
OUTPUT_PATH = RESULTS_DIR / "delta_f1_retained.json"
MODELS = {"qwen2.5-7b": "qwen2.5:7b", "llama3.1-8b": "llama3.1:8b", "mistral-7b": "mistral:7b"}
LANGS = ["pt", "en"]
RUN_PATTERNS = {
    "baseline": "baseline_ollama-{slug}_{lang}_n625_*",
    "pipeline": "pipeline_ollama-{slug}-ollama-{slug}_{lang}_n625_*",
    "two_call": "ablation_*/two_call_baseline_ollama-{slug}_{lang}_n625_*",
}


def load_predictions(strategy: str, slug: str, lang: str) -> list[dict]:
    pattern = RUN_PATTERNS[strategy].format(slug=slug, lang=lang)
    run_dir = sorted(glob.glob(str(RESULTS_DIR / pattern)))[0]
    data = json.loads(Path(run_dir, "results.json").read_text(encoding="utf-8"))
    return data["predictions"]


def macro_f1(predictions: list[dict], exclude_zero_conf: bool) -> float:
    kept = [p for p in predictions if not (exclude_zero_conf and p["confidence"] == 0.0)]
    y_true = [p["metadata"]["label_type"].upper() for p in kept]
    y_pred = [p["requirement_type"].upper() for p in kept]
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def delta_row(slug: str, lang: str) -> dict:
    runs = {strategy: load_predictions(strategy, slug, lang) for strategy in RUN_PATTERNS}
    row: dict = {"model": MODELS[slug], "lang": lang, "n": len(runs["baseline"])}
    for policy, exclude in (("excluded", True), ("retained", False)):
        f1_baseline = macro_f1(runs["baseline"], exclude)
        f1_two_call = macro_f1(runs["two_call"], exclude)
        f1_pipeline = macro_f1(runs["pipeline"], exclude)
        row[policy] = {
            "f1_baseline": round(f1_baseline, 6),
            "f1_two_call": round(f1_two_call, 6),
            "f1_pipeline": round(f1_pipeline, 6),
            "delta_p": round(f1_two_call - f1_baseline, 6),
            "delta_s": round(f1_pipeline - f1_two_call, 6),
        }
    return row


def main() -> None:
    rows = [delta_row(slug, lang) for slug in MODELS for lang in LANGS]

    print(f"{'model':<14} {'L':<3} {'dp_excl':>8} {'dp_ret':>8} {'ds_excl':>8} {'ds_ret':>8}")
    for r in rows:
        x, t = r["excluded"], r["retained"]
        print(
            f"{r['model']:<14} {r['lang'].upper():<3} {x['delta_p']:>+8.4f} {t['delta_p']:>+8.4f} "
            f"{x['delta_s']:>+8.4f} {t['delta_s']:>+8.4f}"
        )

    OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSalvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
