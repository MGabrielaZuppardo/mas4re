"""
MAS4RE — Reanálise suplementar: GPT-4.1-mini (Azure AI Foundry) vs. painel SLM

O grid reportado no artigo usa 3 SLMs locais (qwen2.5:7b, llama3.1:8b,
mistral:7b). Já existiam no repositório runs de baseline e pipeline (n=625,
mesma configuração) com um modelo proprietário mais forte,
foundry/gpt-4.1-mini, que nunca foram incluídos na grade de 18 condições
nem analisados no artigo. Não existe ablação two-call para este modelo,
então Δp/Δs não podem ser decompostos aqui — este script cobre apenas a
comparação pipeline vs. baseline (equivalente a RQ1) e a taxa de falha de
parsing, como evidência exploratória sobre se o padrão observado no painel
SLM se sustenta para um modelo acima do limiar de compliance.

Uso (a partir de D:/mas4re):
    python experiments/compute_foundry_supplementary.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from experiments.compute_stats import RESULTS_DIR, correct_binary

OUTPUT_PATH = RESULTS_DIR / "foundry_gpt41mini_supplementary.json"

RUN_DIRS = {
    ("baseline", "pt"): "baseline_foundry-gpt-4.1-mini_pt_n625_1781994839",
    ("baseline", "en"): "baseline_foundry-gpt-4.1-mini_en_n625_1781995910",
    ("pipeline", "pt"): "pipeline_foundry-gpt-4.1-mini-foundry-gpt-4.1-mini_pt_n625_1781995175",
    ("pipeline", "en"): "pipeline_foundry-gpt-4.1-mini-foundry-gpt-4.1-mini_en_n625_1781996303",
}


def cohens_h(p1: float, p2: float) -> float:
    def clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    return 2 * math.asin(math.sqrt(clamp(p1))) - 2 * math.asin(math.sqrt(clamp(p2)))


def load_preds(run_dir: str) -> list[dict]:
    path = RESULTS_DIR / run_dir / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))["predictions"]


def metrics(preds: list[dict]) -> dict:
    y_true, y_pred = [], []
    n_zero = 0
    for p in preds:
        gt = p.get("metadata", {}).get("label_type", "").upper()
        rt = (p.get("requirement_type") or "").upper()
        conf = p.get("confidence")
        if conf == 0.0:
            n_zero += 1
        if not gt or not rt:
            continue
        y_true.append(gt)
        y_pred.append(rt)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        mcc = 0.0
    return {"acc": acc, "f1_macro": f1, "mcc": mcc, "zero_conf": n_zero, "n": len(preds)}


def main() -> None:
    print(f"\n{'=' * 88}")
    print("Suplementar: foundry/gpt-4.1-mini — pipeline vs. baseline, fora do grid de 18")
    print("=" * 88)

    output = []
    for lang in ["pt", "en"]:
        base_preds = load_preds(RUN_DIRS[("baseline", lang)])
        pipe_preds = load_preds(RUN_DIRS[("pipeline", lang)])
        base_m = metrics(base_preds)
        pipe_m = metrics(pipe_preds)
        h = cohens_h(pipe_m["acc"], base_m["acc"])

        base_dict = {p["text"]: p for p in base_preds}
        pipe_dict = {p["text"]: p for p in pipe_preds}
        common = sorted(set(base_dict) & set(pipe_dict))
        pairs = [
            (correct_binary(base_dict[t]), correct_binary(pipe_dict[t]))
            for t in common
            if correct_binary(base_dict[t]) is not None and correct_binary(pipe_dict[t]) is not None
        ]
        bs = [p[0] for p in pairs]
        ps = [p[1] for p in pairs]
        stat, p_value = stats.wilcoxon(ps, bs, alternative="two-sided", zero_method="wilcox")

        output.append(
            {
                "lang": lang,
                "baseline": base_m,
                "pipeline": pipe_m,
                "delta_acc": pipe_m["acc"] - base_m["acc"],
                "delta_f1_macro": pipe_m["f1_macro"] - base_m["f1_macro"],
                "cohens_h": round(h, 4),
                "n_paired": len(pairs),
                "wilcoxon_p": float(p_value),
            }
        )
        print(f"\n  {lang.upper()}:")
        print(
            f"    baseline  acc={base_m['acc']:.4f}  f1={base_m['f1_macro']:.4f}  "
            f"mcc={base_m['mcc']:.4f}  zero_conf={base_m['zero_conf']}/{base_m['n']}"
        )
        print(
            f"    pipeline  acc={pipe_m['acc']:.4f}  f1={pipe_m['f1_macro']:.4f}  "
            f"mcc={pipe_m['mcc']:.4f}  zero_conf={pipe_m['zero_conf']}/{pipe_m['n']}"
        )
        print(
            f"    Δacc={pipe_m['acc'] - base_m['acc']:+.4f}  "
            f"Δf1={pipe_m['f1_macro'] - base_m['f1_macro']:+.4f}  "
            f"h={h:+.4f}  Wilcoxon(n={len(pairs)}) p={p_value:.4g}"
        )

    print(
        "\nAviso: nao ha ablacao two-call para este modelo neste repositorio; "
        "Delta_p/Delta_s nao podem ser decompostos aqui. Analise exploratoria/"
        "suplementar, fora da grade de 18 condicoes reportada no artigo."
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
