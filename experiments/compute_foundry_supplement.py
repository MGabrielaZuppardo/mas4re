"""
MAS4RE — Suplemento gpt-4.1-mini (Azure AI Foundry): baseline, two-call e pipeline

Reúne as seis condições (3 arquiteturas x PT/EN) e calcula, por idioma, acurácia e F1
macro e as três comparações pareadas por requisito: pipeline - baseline (RQ1),
two-call - baseline (Delta_p) e pipeline - two-call (Delta_s). Teste principal: McNemar
exato sobre os pares discordantes; Wilcoxon pareado para continuidade com o rebuttal.
Effect size: Cohen's g pareado.

Entradas (todas com n=625, sem falhas de classificação por confiança zero):
  - EN baseline e pipeline: experiments/results/foundry_en_rerun (agentes com text_en);
  - PT e EN two-call: runs do suplemento em experiments/results/.

Uso (a partir da raiz do repositório):
    python experiments/compute_foundry_supplement.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score, f1_score

RESULTS = Path("experiments/results")
OUTPUT_PATH = RESULTS / "foundry_supplement.json"
PATTERNS = {
    ("baseline", "en"): "foundry_en_rerun/baseline_foundry-gpt-4.1-mini_en_n625_*",
    (
        "pipeline",
        "en",
    ): "foundry_en_rerun/pipeline_foundry-gpt-4.1-mini-foundry-gpt-4.1-mini_en_n625_*",
    ("two_call", "en"): "two_call_baseline_foundry-gpt-4.1-mini_en_n625_*",
    ("baseline", "pt"): "baseline_foundry-gpt-4.1-mini_pt_n625_*",
    ("pipeline", "pt"): "pipeline_foundry-gpt-4.1-mini-foundry-gpt-4.1-mini_pt_n625_*",
    ("two_call", "pt"): "two_call_baseline_foundry-gpt-4.1-mini_pt_n625_*",
}


def load_run(pattern: str) -> dict[str, dict]:
    run_dir = sorted(glob.glob(str(RESULTS / pattern)))[0]
    data = json.loads(Path(run_dir, "results.json").read_text(encoding="utf-8"))
    return {p["text"]: p for p in data["predictions"]}


def _correct(prediction: dict) -> bool:
    return prediction["metadata"]["label_type"].upper() == prediction["requirement_type"].upper()


def _scores(run: dict[str, dict]) -> dict:
    labels = [
        (p["metadata"]["label_type"].upper(), p["requirement_type"].upper()) for p in run.values()
    ]
    y_true, y_pred = [a for a, _ in labels], [b for _, b in labels]
    return {
        "n": len(labels),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "classification_fallbacks": sum(p["confidence"] == 0.0 for p in run.values()),
        "priority_fallbacks": sum(
            (p.get("priority_justification") or "").startswith("Parse falhou") for p in run.values()
        ),
    }


def compare(first: dict, second: dict) -> dict:
    texts = sorted(set(first) & set(second))
    a = [_correct(first[t]) for t in texts]
    b_ = [_correct(second[t]) for t in texts]
    only_first = sum(x and not y for x, y in zip(a, b_))
    only_second = sum(y and not x for x, y in zip(a, b_))
    discordant = only_first + only_second
    diffs = [int(x) - int(y) for x, y in zip(a, b_)]
    return {
        "n_paired": len(texts),
        "delta_accuracy": round(sum(a) / len(a) - sum(b_) / len(b_), 4),
        "first_only_correct": only_first,
        "second_only_correct": only_second,
        "mcnemar_exact_p": float(binomtest(only_first, discordant, 0.5).pvalue)
        if discordant
        else 1.0,
        "wilcoxon_p": float(wilcoxon(diffs).pvalue) if any(diffs) else 1.0,
        "cohens_g": round(only_first / discordant - 0.5, 4) if discordant else 0.0,
    }


def main() -> None:
    runs = {key: load_run(pattern) for key, pattern in PATTERNS.items()}
    report: dict = {}
    for lang in ("pt", "en"):
        base, two, pipe = (
            runs[("baseline", lang)],
            runs[("two_call", lang)],
            runs[("pipeline", lang)],
        )
        report[lang] = {
            "scores": {
                "baseline": _scores(base),
                "two_call": _scores(two),
                "pipeline": _scores(pipe),
            },
            "pipeline_vs_baseline": compare(pipe, base),
            "delta_p_two_call_vs_baseline": compare(two, base),
            "delta_s_pipeline_vs_two_call": compare(pipe, two),
        }
        print(f"\n== {lang.upper()}")
        for name, scores in report[lang]["scores"].items():
            print(f"  {name:9s} {scores}")
        for key in (
            "pipeline_vs_baseline",
            "delta_p_two_call_vs_baseline",
            "delta_s_pipeline_vs_two_call",
        ):
            print(f"  {key:30s} {report[lang][key]}")
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
