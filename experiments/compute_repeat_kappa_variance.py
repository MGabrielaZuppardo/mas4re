"""
MAS4RE — Variação do Fleiss' kappa entre execuções idênticas

O bootstrap de compute_rq3_kappa_bootstrap.py controla apenas a reamostragem dos
requisitos; não captura o não-determinismo do servidor de modelos a temperature=0.
Este script usa a repetição de run_repeat_grid.py (mesmo código, seed e configuração)
para comparar, por idioma:

  - kappa de cada arquitetura (pipeline, two_call_baseline) em cada execução;
  - Delta-kappa (pipeline - two_call) em cada execução e nos pareamentos cruzados;
  - ruído de reexecução: kappa de uma mesma arquitetura entre as duas execuções.

Se o Delta-kappa não excede o ruído de reexecução, ele não pode ser atribuído à
arquitetura. Os requisitos são os alinhados por texto em todas as 12 combinações
(arquitetura x execução x modelo), de modo que todos os kappas usam o mesmo n.

Uso (a partir da raiz do repositório):
    python experiments/compute_repeat_kappa_variance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.compute_stats import (
    RESULTS_DIR,
    _find_latest_csv,
    fleiss_kappa,
    load_all_runs,
    load_csv,
    load_two_call_runs,
)

MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
LANGS = ["pt", "en"]
ARCHITECTURES = ("pipeline", "two_call_baseline")
REPEAT_DIR = RESULTS_DIR / "repeat_run2"
OUTPUT_PATH = RESULTS_DIR / "repeat_kappa_variance.json"
N_BOOT = 5_000
SEED = 42


def _short_model(config_model: str) -> str:
    return config_model.split("-ollama/")[0].removeprefix("ollama/")


def load_repeat_runs(repeat_dir: Path) -> dict[tuple, dict]:
    runs: dict[tuple, dict] = {}
    for results_path in sorted(repeat_dir.glob("*/results.json")):
        manifest = json.loads((results_path.parent / "manifest.json").read_text(encoding="utf-8"))
        data = json.loads(results_path.read_text(encoding="utf-8"))
        key = (manifest["strategy"], _short_model(manifest["model"]), manifest["lang"])
        runs[key] = {p["text"]: p for p in data["predictions"]}
    return runs


def _priorities(runs: dict, architecture: str, lang: str) -> dict[str, dict[str, str]]:
    return {
        model: {
            text: p.get("priority", "C") for text, p in runs[(architecture, model, lang)].items()
        }
        for model in MODELS
    }


def _aligned_texts(priorities_by_set: list[dict[str, dict[str, str]]]) -> list[str]:
    texts = set.intersection(*(set(by_model[m]) for by_model in priorities_by_set for m in MODELS))
    return sorted(texts)


def _matrix(by_model: dict[str, dict[str, str]], texts: list[str]) -> list[list[str]]:
    return [[by_model[model][text] for model in MODELS] for text in texts]


def _kappas(
    matrices: dict[str, list[list[str]]], idx: np.ndarray | None = None
) -> dict[str, float]:
    pick = (lambda m: m) if idx is None else (lambda m: [m[i] for i in idx])
    return {name: float(fleiss_kappa(pick(matrix))) for name, matrix in matrices.items()}


def _contrasts(k: dict[str, float]) -> dict[str, float]:
    return {
        "delta_run1": k["pipeline_run1"] - k["two_call_run1"],
        "delta_run2": k["pipeline_run2"] - k["two_call_run2"],
        "delta_cross_a": k["pipeline_run1"] - k["two_call_run2"],
        "delta_cross_b": k["pipeline_run2"] - k["two_call_run1"],
        "noise_pipeline": k["pipeline_run1"] - k["pipeline_run2"],
        "noise_two_call": k["two_call_run1"] - k["two_call_run2"],
    }


def _percentile_ci(samples: np.ndarray) -> list[float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def analyse_language(original: dict, repeat: dict, lang: str, rng: np.random.Generator) -> dict:
    sets = {
        "pipeline_run1": _priorities(original, "pipeline", lang),
        "two_call_run1": _priorities(original, "two_call_baseline", lang),
        "pipeline_run2": _priorities(repeat, "pipeline", lang),
        "two_call_run2": _priorities(repeat, "two_call_baseline", lang),
    }
    texts = _aligned_texts(list(sets.values()))
    matrices = {name: _matrix(by_model, texts) for name, by_model in sets.items()}

    observed_kappas = _kappas(matrices)
    observed = _contrasts(observed_kappas)

    boot = {name: np.empty(N_BOOT) for name in observed}
    for b in range(N_BOOT):
        resampled = _contrasts(_kappas(matrices, rng.integers(0, len(texts), size=len(texts))))
        for name, value in resampled.items():
            boot[name][b] = value

    deltas = [observed[k] for k in ("delta_run1", "delta_run2", "delta_cross_a", "delta_cross_b")]
    noise = [abs(observed["noise_pipeline"]), abs(observed["noise_two_call"])]
    return {
        "n_aligned": len(texts),
        "kappa": {name: round(value, 4) for name, value in observed_kappas.items()},
        "contrasts": {name: round(value, 4) for name, value in observed.items()},
        "contrasts_ci95_bootstrap": {name: _percentile_ci(s) for name, s in boot.items()},
        "delta_range": [round(min(deltas), 4), round(max(deltas), 4)],
        "max_abs_rerun_noise": round(max(noise), 4),
        "delta_exceeds_rerun_noise": bool(min(deltas) > max(noise)),
    }


def main() -> None:
    original = load_all_runs(load_csv(_find_latest_csv()))
    original.update(load_two_call_runs())
    repeat = load_repeat_runs(REPEAT_DIR)

    missing = [
        (a, m, lang)
        for a in ARCHITECTURES
        for m in MODELS
        for lang in LANGS
        if (a, m, lang) not in repeat
    ]
    if missing:
        raise SystemExit(f"Repetição incompleta, faltam {len(missing)} condições: {missing}")

    rng = np.random.default_rng(SEED)
    report = {lang: analyse_language(original, repeat, lang, rng) for lang in LANGS}

    for lang, result in report.items():
        k, c = result["kappa"], result["contrasts"]
        print(f"\n== {lang.upper()} (n={result['n_aligned']})")
        print(f"  kappa pipeline  : run1={k['pipeline_run1']:+.4f}  run2={k['pipeline_run2']:+.4f}")
        print(f"  kappa two-call  : run1={k['two_call_run1']:+.4f}  run2={k['two_call_run2']:+.4f}")
        print(f"  delta-kappa     : run1={c['delta_run1']:+.4f}  run2={c['delta_run2']:+.4f}")
        print(f"  deltas cruzados : {c['delta_cross_a']:+.4f}  {c['delta_cross_b']:+.4f}")
        print(
            f"  ruido (mesma arquitetura, run1-run2): pipeline={c['noise_pipeline']:+.4f}  "
            f"two-call={c['noise_two_call']:+.4f}"
        )
        print(f"  delta excede o ruido de reexecucao? {result['delta_exceeds_rerun_noise']}")

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
