"""
MAS4RE — Imprime as tabelas do artigo a partir dos resultados versionados

Lê os JSONs de experiments/results/ (gerados pelos scripts compute_*.py) e as predições
das execuções, e imprime, no layout do artigo, as tabelas de resultados. Não executa
nenhum experimento nem recalcula testes estatísticos, exceto Acc/F1/MCC por condição,
derivados das predições.

Uso (a partir da raiz do repositório):
    python experiments/print_paper_tables.py
    python experiments/print_paper_tables.py --only kappa
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

RESULTS = Path("experiments/results")
MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
SLUGS = {"qwen2.5:7b": "qwen2.5-7b", "llama3.1:8b": "llama3.1-8b", "mistral:7b": "mistral-7b"}


def read_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def predictions(pattern: str) -> list[dict]:
    run_dir = sorted(glob.glob(str(RESULTS / pattern)))[0]
    return json.loads(Path(run_dir, "results.json").read_text(encoding="utf-8"))["predictions"]


def scores(preds: list[dict]) -> tuple[float, float, float]:
    kept = [p for p in preds if p["confidence"] != 0.0]
    y_true = [p["metadata"]["label_type"].upper() for p in kept]
    y_pred = [p["requirement_type"].upper() for p in kept]
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        matthews_corrcoef(y_true, y_pred),
    )


def fmt_p(value: float) -> str:
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def render(heading: str, headers: list[str], rows: list[list[str]], text_columns: int = 2) -> None:
    """Imprime uma tabela com colunas de texto à esquerda e numéricas à direita."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  ".join(
            c.ljust(w) if i < text_columns else c.rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths))
        )

    print(f"\n{'=' * 100}\n{heading}\n{'=' * 100}")
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))


def table_rq1() -> None:
    grid = read_json("statistical_results.json")["rq1_pipeline_vs_baseline"]
    mcnemar = {
        (r["model"], r["lang"]): r
        for r in read_json("statistical_results_mcnemar.json")["rq1_pipeline_vs_baseline_mcnemar"]
    }
    effect = {
        (r["model"], r["lang"]): r for r in read_json("cohens_g_paired.json")["rq1_table4_cohens_g"]
    }
    rows = []
    for item in grid:
        key = (item["model"], item["lang"])
        slug = SLUGS[item["model"]]
        base = scores(predictions(f"baseline_ollama-{slug}_{item['lang']}_n625_*"))
        pipe = scores(predictions(f"pipeline_ollama-{slug}-ollama-{slug}_{item['lang']}_n625_*"))
        g = effect[key]
        rows.append(
            [
                item["model"],
                item["lang"].upper(),
                f"{base[0]:.3f}/{base[1]:.3f}/{base[2]:.3f}",
                f"{pipe[0]:.3f}/{pipe[1]:.3f}/{pipe[2]:.3f}",
                signed(item["cohens_h"]),
                fmt_p(item["p_value"]),
                fmt_p(mcnemar[key]["p_value"]),
                signed(g["cohens_g"]) + ("" if g["verdict"].startswith("efeito") else " (†)"),
            ]
        )
    render(
        "Table 4 - RQ1, pipeline vs baseline (p_W: Wilcoxon, fallbacks excluded; p_M, g: "
        "McNemar and paired g on 623 pairs, retained; † = g not estimable)",
        ["model", "L", "baseline Acc/F1/MCC", "pipeline Acc/F1/MCC", "h", "p_W", "p_M", "g"],
        rows,
    )


def table_language() -> None:
    unpaired = read_json("statistical_results.json")["rq2_language_en_vs_pt"]
    paired = {
        (r["strategy"], r["model"]): r["p_value"]
        for r in read_json("statistical_results_mcnemar.json")["language_effect_mcnemar_paired"]
    }
    rows = [
        [
            r["strategy"],
            r["model"],
            f"{r['pt_acc']:.3f}",
            f"{r['en_acc']:.3f}",
            signed(r["cohens_h"]),
            fmt_p(r["p_value"]),
            fmt_p(paired[(r["strategy"], r["model"])]),
        ]
        for r in unpaired
    ]
    render(
        "Table 5 - language effect (PT vs EN): unpaired Mann-Whitney vs paired exact McNemar",
        ["architecture", "model", "PT acc", "EN acc", "h", "p_MW", "p_McNemar"],
        rows,
    )


def table_kappa() -> None:
    kappa = read_json("kappa_table.json")
    rows = [
        [
            name,
            str(row["n_aligned"]),
            f"{row['kappa']:+.4f}",
            row["interpretation"],
            "/".join(f"{row['must_rate'][m]:.2f}" for m in MODELS),
        ]
        for name, row in kappa.items()
    ]
    render(
        "Table 6 - inter-model MoSCoW consistency (Fleiss kappa, 3 raters) and Must rate "
        "(qwen/llama/mistral)",
        ["stratum", "n", "kappa", "interpretation", "Must rate"],
        rows,
        text_columns=1,
    )
    latest = sorted(RESULTS.glob("rq3_kappa_bootstrap_*.json"))[-1]
    comparisons = json.loads(latest.read_text(encoding="utf-8"))["comparisons"]
    rows = [
        [
            c["lang"].upper(),
            f"{c['delta_kappa']:+.4f}",
            f"[{c['ci_bonferroni_low']:+.4f}, {c['ci_bonferroni_high']:+.4f}]",
            fmt_p(c["p_bootstrap_two_sided"]),
        ]
        for c in comparisons
    ]
    render(
        "Bootstrap of the kappa increment (pipeline - two-call), Bonferroni CI over 2 languages",
        ["L", "delta_kappa", "97.5% CI", "p"],
        rows,
        text_columns=1,
    )


def table_decomposition() -> None:
    effect = {
        (r["model"], r["lang"]): r
        for r in read_json("cohens_g_paired.json")["delta_state_table9_cohens_g"]
    }
    rows = []
    for item in read_json("delta_f1_retained.json"):
        x, t = item["excluded"], item["retained"]
        g = effect[(item["model"], item["lang"])]
        rows.append(
            [
                item["model"],
                item["lang"].upper(),
                f"{x['f1_baseline']:.3f}/{x['f1_two_call']:.3f}/{x['f1_pipeline']:.3f}",
                signed(x["delta_p"]),
                signed(x["delta_s"]),
                signed(t["delta_p"]),
                signed(t["delta_s"]),
                f"{g['pipe_only_correct']}/{g['two_call_only_correct']}",
                signed(g["cohens_g"]),
            ]
        )
    render(
        "Table 9 - decomposition ablation, F1 macro (excluded vs retained fallbacks); "
        "b/c and g refer to delta_s",
        ["model", "L", "Base/2Call/Pipe", "dp excl", "ds excl", "dp ret", "ds ret", "b/c", "g"],
        rows,
    )


def table_baseline_vs_two_call() -> None:
    rows = []
    for item in read_json("baseline_vs_two_call.json"):
        for policy in ("excluded", "retained"):
            r = item[policy]
            rows.append(
                [
                    item["model"],
                    item["lang"].upper(),
                    policy,
                    str(r["n"]),
                    signed(r["cohens_h"]),
                    signed(r["cohens_g"]),
                    fmt_p(r["mcnemar_exact_p"]),
                    "yes" if r["significant_mcnemar_bonferroni"] else "no",
                ]
            )
    render(
        "Section 6.5 - baseline vs two-call (decomposition alone), exact McNemar, "
        "Bonferroni alpha' = 0.0083",
        ["model", "L", "policy", "n", "h", "g", "p_McNemar", "significant"],
        rows,
        text_columns=3,
    )


def table_foundry() -> None:
    data = read_json("foundry_supplement.json")
    rows = []
    for lang in LANGS:
        accuracy = {k: v["accuracy"] for k, v in data[lang]["scores"].items()}
        for key, label in (
            ("pipeline_vs_baseline", "pipeline - baseline"),
            ("delta_p_two_call_vs_baseline", "delta_p"),
            ("delta_s_pipeline_vs_two_call", "delta_s"),
        ):
            c = data[lang][key]
            rows.append(
                [
                    lang.upper(),
                    label,
                    f"{accuracy['baseline']:.3f}/{accuracy['two_call']:.3f}/{accuracy['pipeline']:.3f}",
                    signed(c["delta_accuracy"], 4),
                    f"{c['first_only_correct']}/{c['second_only_correct']}",
                    fmt_p(c["mcnemar_exact_p"]),
                    signed(c["cohens_g"]),
                ]
            )
    render(
        "Supplement - gpt-4.1-mini (accuracy; 623 paired requirements; exact McNemar)",
        ["L", "comparison", "base/2call/pipe", "delta acc", "b/c", "p_McNemar", "g"],
        rows,
        text_columns=2,
    )


def table_repeat() -> None:
    data = read_json("repeat_kappa_variance.json")
    rows = []
    for lang in LANGS:
        k, c = data[lang]["kappa"], data[lang]["contrasts"]
        rows.append(
            [
                lang.upper(),
                f"{k['pipeline_run1']:+.4f}/{k['pipeline_run2']:+.4f}",
                f"{k['two_call_run1']:+.4f}/{k['two_call_run2']:+.4f}",
                f"{c['delta_run1']:+.4f}/{c['delta_run2']:+.4f}",
                f"{c['delta_cross_a']:+.4f}/{c['delta_cross_b']:+.4f}",
                f"{data[lang]['max_abs_rerun_noise']:.4f}",
            ]
        )
    render(
        "Repeat execution - kappa and delta_kappa in two identical executions (run1/run2)",
        ["L", "kappa pipeline", "kappa two-call", "delta_kappa", "cross pairings", "max noise"],
        rows,
        text_columns=1,
    )


TABLES = {
    "rq1": table_rq1,
    "language": table_language,
    "kappa": table_kappa,
    "decomposition": table_decomposition,
    "baseline_vs_two_call": table_baseline_vs_two_call,
    "foundry": table_foundry,
    "repeat": table_repeat,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Imprime as tabelas do artigo")
    parser.add_argument("--only", choices=list(TABLES), help="Imprime só uma tabela")
    args = parser.parse_args()
    for name, render_table in TABLES.items():
        if args.only in (None, name):
            render_table()


if __name__ == "__main__":
    main()
