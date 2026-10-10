"""
MAS4RE — Concordância item a item entre pipeline e two-call baseline

O pipeline e o two-call usam os mesmos prompts, e a validação do estado tipado não
rejeitou nenhuma saída real; ainda assim suas saídas não são idênticas. Este script
mede, por condição, em que fração dos requisitos os dois braços produzem o mesmo tipo,
a mesma categoria e a mesma prioridade, para dimensionar o quanto diferenças de execução
(ordem de chamadas, não-determinismo do servidor) afastam as duas arquiteturas.

Considera os requisitos alinhados por texto, sem fallback de parsing em nenhum braço.
Saída: experiments/results/pipeline_vs_two_call_agreement.json.

Uso (a partir da raiz do repositório):
    python experiments/compute_pipeline_vs_two_call_agreement.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
OUTPUT_PATH = RESULTS_DIR / "pipeline_vs_two_call_agreement.json"
MODELS = {"qwen2.5:7b": "qwen2.5-7b", "llama3.1:8b": "llama3.1-8b", "mistral:7b": "mistral-7b"}
LANGS = ["pt", "en"]


def load_run(pattern: str) -> dict[str, dict]:
    run_dir = sorted(glob.glob(str(RESULTS_DIR / pattern)))[0]
    data = json.loads(Path(run_dir, "results.json").read_text(encoding="utf-8"))
    return {p["text"]: p for p in data["predictions"]}


def agreement(pipeline: dict[str, dict], two_call: dict[str, dict]) -> dict:
    texts = [
        t
        for t in sorted(set(pipeline) & set(two_call))
        if pipeline[t]["confidence"] != 0.0 and two_call[t]["confidence"] != 0.0
    ]

    def same(field: str) -> float:
        return sum(pipeline[t].get(field) == two_call[t].get(field) for t in texts) / len(texts)

    return {
        "n": len(texts),
        "same_type": round(same("requirement_type"), 4),
        "same_category": round(same("nfr_category"), 4),
        "same_priority": round(same("priority"), 4),
        "type_disagreement": round(1 - same("requirement_type"), 4),
        "priority_disagreement": round(1 - same("priority"), 4),
    }


def main() -> None:
    rows = []
    for model, slug in MODELS.items():
        for lang in LANGS:
            pipeline = load_run(f"pipeline_ollama-{slug}-ollama-{slug}_{lang}_n625_*")
            two_call = load_run(f"ablation_*/two_call_baseline_ollama-{slug}_{lang}_n625_*")
            rows.append({"model": model, "lang": lang, **agreement(pipeline, two_call)})

    print(f"{'model':<14}{'L':<4}{'n':>5}{'type':>8}{'category':>10}{'priority':>10}")
    for r in rows:
        print(
            f"{r['model']:<14}{r['lang'].upper():<4}{r['n']:>5}{r['same_type']:>8.3f}"
            f"{r['same_category']:>10.3f}{r['same_priority']:>10.3f}"
        )
    summary = {
        "max_type_disagreement": max(r["type_disagreement"] for r in rows),
        "max_priority_disagreement": max(r["priority_disagreement"] for r in rows),
    }
    print(
        f"\nmax type disagreement: {summary['max_type_disagreement']:.3f} | "
        f"max priority disagreement: {summary['max_priority_disagreement']:.3f}"
    )

    OUTPUT_PATH.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"Salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
