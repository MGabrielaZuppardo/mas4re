"""Would an independent classical verifier predict the LLM's F/NF errors? (EXPLORATORY)

Trains a TF-IDF + logistic-regression classifier with out-of-fold predictions grouped by
project (no project leaks between train and test) and checks whether its confident
disagreement with the LLM flags LLM errors. It is a pseudo-oracle (a second, independent
implementation of the same function).

HYBRID: the verifier is trained on labelled data, so this analysis is NOT part of the
zero-shot main study. Its manifest is stamped information_regime=hybrid_exploratory so it is
never mixed with zero-shot results.

Usage (from the repository root):
    python -m experiments.analyze_verifier_oracle
    python -m experiments.analyze_verifier_oracle --min-confidence 0.90

Output:
    experiments/results/analysis/verifier_oracle_<timestamp>.json
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from config.settings import settings
from datasets.promise import PromiseAdapter
from domain.enums import InformationRegime
from experiments.analysis_common import (
    FULL_DATASET_N,
    GENERATIONS,
    HISTORICAL,
    build_analysis_manifest,
    format_percent,
    is_parse_failure,
    latest_run_per_model_and_lang,
    load_full_runs,
    ratio,
    write_analysis_result,
)

N_SPLITS = 5
DEFAULT_MIN_CONFIDENCE = 0.85


def out_of_fold_nfr_probability(
    texts: list[str], is_nfr: np.ndarray, groups: np.ndarray, n_splits: int = N_SPLITS
) -> np.ndarray:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0),
    )
    probability = np.zeros(len(is_nfr))
    for train, test in GroupKFold(n_splits=n_splits).split(texts, is_nfr, groups):
        model.fit([texts[i] for i in train], is_nfr[train])
        probability[test] = model.predict_proba([texts[i] for i in test])[:, 1]
    return probability


def verifier_report(
    llm_is_nfr: np.ndarray,
    truth_is_nfr: np.ndarray,
    verifier_is_nfr: np.ndarray,
    verifier_confidence: np.ndarray,
    min_confidence: float,
) -> dict[str, float | int | None]:
    llm_wrong = llm_is_nfr != truth_is_nfr
    disagrees = llm_is_nfr != verifier_is_nfr
    confident_disagreement = disagrees & (verifier_confidence >= min_confidence)
    overridden = np.where(confident_disagreement, verifier_is_nfr, llm_is_nfr)
    return {
        "n": len(truth_is_nfr),
        "llm_accuracy": accuracy_score(truth_is_nfr, llm_is_nfr),
        "disagreement_rate": float(disagrees.mean()),
        "llm_error_rate_when_disagree": ratio(llm_wrong[disagrees].sum(), disagrees.sum()),
        "llm_errors_captured": ratio((llm_wrong & disagrees).sum(), llm_wrong.sum()),
        "confident_disagreements": int(confident_disagreement.sum()),
        "llm_error_rate_when_confident": ratio(
            llm_wrong[confident_disagreement].sum(), confident_disagreement.sum()
        ),
        "accuracy_gain_if_verifier_overrides": accuracy_score(truth_is_nfr, overridden)
        - accuracy_score(truth_is_nfr, llm_is_nfr),
    }


def _report_for_run(
    predictions: list[dict[str, Any]],
    index_by_text: dict[str, int],
    is_nfr: np.ndarray,
    probability: np.ndarray,
    min_confidence: float,
) -> dict[str, float | int | None]:
    kept = [p for p in predictions if not is_parse_failure(p) and p["text"] in index_by_text]
    rows = np.array([index_by_text[p["text"]] for p in kept])
    llm_is_nfr = np.array([p["requirement_type"] == "NF" for p in kept])
    p_nfr = probability[rows]
    return verifier_report(
        llm_is_nfr,
        is_nfr[rows],
        p_nfr >= 0.5,
        np.maximum(p_nfr, 1 - p_nfr),
        min_confidence,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", default="pipeline")
    parser.add_argument("--n", type=int, default=FULL_DATASET_N)
    parser.add_argument("--generation", choices=list(GENERATIONS), default=HISTORICAL)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    requirements = PromiseAdapter(path=settings.promise_dataset_path).load()
    texts = [r.text_en or r.text for r in requirements]
    is_nfr = np.array([r.metadata["label_type"] == "NF" for r in requirements])
    groups = np.array([r.metadata["project"] for r in requirements])
    index_by_text = {r.text: i for i, r in enumerate(requirements)}

    probability = out_of_fold_nfr_probability(texts, is_nfr, groups)
    verifier_is_nfr = probability >= 0.5
    verifier = {
        "accuracy": accuracy_score(is_nfr, verifier_is_nfr),
        "f1_macro": f1_score(is_nfr, verifier_is_nfr, average="macro"),
        "majority_class_share": float(max(is_nfr.mean(), 1 - is_nfr.mean())),
        "projects": len(set(groups)),
    }
    print(f"verifier (leave-project-out CV): {verifier}\n")

    runs = load_full_runs(strategy=args.strategy, n=args.n, generation=args.generation)
    reports: dict[str, dict[str, float | int | None]] = {}
    for lang, models in latest_run_per_model_and_lang(runs).items():
        for model, run in models.items():
            key = f"{model}/{lang}"
            reports[key] = _report_for_run(
                run.predictions, index_by_text, is_nfr, probability, args.min_confidence
            )
            r = reports[key]
            print(
                f"{key:<32} acc={r['llm_accuracy']:.3f} "
                f"confident={r['confident_disagreements']:>4} "
                f"llm_err|confident={format_percent(r['llm_error_rate_when_confident']):>6} "
                f"gain={r['accuracy_gain_if_verifier_overrides']:+.3f}"
            )

    manifest = build_analysis_manifest(
        "analyze_verifier_oracle",
        InformationRegime.HYBRID_EXPLORATORY,
        {
            "strategy": args.strategy,
            "n": args.n,
            "generation": args.generation,
            "min_confidence": args.min_confidence,
            "n_splits": N_SPLITS,
            "runs_loaded": len(runs),
        },
    )
    path = write_analysis_result(
        "verifier_oracle", manifest, {"verifier": verifier, "per_run": reports}
    )
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
