"""
MAS4RE — Reanálise suplementar: GPT-4.1-mini (Azure AI Foundry) vs. painel SLM

O grid reportado no artigo usa 3 SLMs locais (qwen2.5:7b, llama3.1:8b,
mistral:7b). Já existiam no repositório runs de baseline e pipeline (n=625,
mesma configuração) com um modelo proprietário mais forte,
foundry/gpt-4.1-mini, que nunca foram incluídos na grade de 18 condições
nem analisados no artigo. O braço two_call_baseline foi executado
posteriormente (scripts/run_two_call_foundry.py) especificamente para
fechar esta lacuna, permitindo decompor Δp/Δs (RQ2) também para este
modelo — evidência de se o padrão Δp≫Δs≈0 do painel SLM se sustenta para
um modelo acima do limiar de compliance.

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
    ("two_call_baseline", "pt"): "two_call_baseline_foundry-gpt-4.1-mini_pt_n625_1789434668",
    ("two_call_baseline", "en"): "two_call_baseline_foundry-gpt-4.1-mini_en_n625_1789435364",
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

    def paired_wilcoxon(a_preds: list[dict], b_preds: list[dict]) -> tuple[int, float]:
        a_dict = {p["text"]: p for p in a_preds}
        b_dict = {p["text"]: p for p in b_preds}
        common = sorted(set(a_dict) & set(b_dict))
        pairs = [
            (correct_binary(a_dict[t]), correct_binary(b_dict[t]))
            for t in common
            if correct_binary(a_dict[t]) is not None and correct_binary(b_dict[t]) is not None
        ]
        a_vals = [p[0] for p in pairs]
        b_vals = [p[1] for p in pairs]
        _, p_value = stats.wilcoxon(b_vals, a_vals, alternative="two-sided", zero_method="wilcox")
        return len(pairs), float(p_value)

    output = []
    for lang in ["pt", "en"]:
        base_preds = load_preds(RUN_DIRS[("baseline", lang)])
        pipe_preds = load_preds(RUN_DIRS[("pipeline", lang)])
        two_preds = load_preds(RUN_DIRS[("two_call_baseline", lang)])
        base_m = metrics(base_preds)
        pipe_m = metrics(pipe_preds)
        two_m = metrics(two_preds)

        n_bp, p_bp = paired_wilcoxon(base_preds, pipe_preds)  # RQ1: pipeline vs baseline
        n_bt, p_bt = paired_wilcoxon(base_preds, two_preds)  # Delta_p: two-call vs baseline
        n_tp, p_tp = paired_wilcoxon(two_preds, pipe_preds)  # Delta_s: pipeline vs two-call

        delta_p = two_m["acc"] - base_m["acc"]
        delta_s = pipe_m["acc"] - two_m["acc"]
        h_rq1 = cohens_h(pipe_m["acc"], base_m["acc"])
        h_p = cohens_h(two_m["acc"], base_m["acc"])
        h_s = cohens_h(pipe_m["acc"], two_m["acc"])

        output.append(
            {
                "lang": lang,
                "baseline": base_m,
                "two_call_baseline": two_m,
                "pipeline": pipe_m,
                "rq1_delta_acc": pipe_m["acc"] - base_m["acc"],
                "rq1_cohens_h": round(h_rq1, 4),
                "rq1_wilcoxon_p": p_bp,
                "delta_p": delta_p,
                "delta_p_cohens_h": round(h_p, 4),
                "delta_p_wilcoxon_p": p_bt,
                "delta_s": delta_s,
                "delta_s_cohens_h": round(h_s, 4),
                "delta_s_wilcoxon_p": p_tp,
                "n_paired": n_bp,
            }
        )
        print(f"\n  {lang.upper()}:")
        print(
            f"    baseline   acc={base_m['acc']:.4f}  f1={base_m['f1_macro']:.4f}  "
            f"zero_conf={base_m['zero_conf']}/{base_m['n']}"
        )
        print(
            f"    two_call   acc={two_m['acc']:.4f}  f1={two_m['f1_macro']:.4f}  "
            f"zero_conf={two_m['zero_conf']}/{two_m['n']}"
        )
        print(
            f"    pipeline   acc={pipe_m['acc']:.4f}  f1={pipe_m['f1_macro']:.4f}  "
            f"zero_conf={pipe_m['zero_conf']}/{pipe_m['n']}"
        )
        print(
            f"    RQ1 (pipe-base)  Δacc={pipe_m['acc'] - base_m['acc']:+.4f}  "
            f"h={h_rq1:+.4f}  Wilcoxon p={p_bp:.4g}"
        )
        print(f"    Δp  (2call-base) Δacc={delta_p:+.4f}  h={h_p:+.4f}  Wilcoxon p={p_bt:.4g}")
        print(f"    Δs  (pipe-2call) Δacc={delta_s:+.4f}  h={h_s:+.4f}  Wilcoxon p={p_tp:.4g}")

    print(
        "\nDecomposicao Delta_p/Delta_s agora disponivel para foundry/gpt-4.1-mini "
        "(braco two_call_baseline executado via scripts/run_two_call_foundry.py). "
        "Ainda uma analise suplementar/exploratoria, fora da grade de 18 condicoes "
        "reportada no artigo."
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
