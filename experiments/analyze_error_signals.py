"""Which signals predict the classifier's F/NF errors? (historical runs, no LLM calls)

For each candidate signal it reports how many items it flags, the error rate among
flagged vs unflagged items and the share of all errors it captures. It also measures
how well disagreement between models predicts errors (N-version style) and the
sampling uncertainty (Wilson 95%) of each run's accuracy.

Usage (from the repository root):
    python -m experiments.analyze_error_signals
    python -m experiments.analyze_error_signals --strategy baseline

Output:
    experiments/results/analysis/error_signals_<timestamp>.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from typing import Any

from domain.enums import InformationRegime
from experiments.analysis_common import (
    FULL_DATASET_N,
    GENERATIONS,
    HISTORICAL,
    Prediction,
    RunData,
    build_analysis_manifest,
    format_percent,
    has_inter_agent_conflict,
    has_structural_inconsistency,
    is_parse_failure,
    is_type_error,
    latest_run_per_model_and_lang,
    load_full_runs,
    ratio,
    required_sample_size,
    wilson_interval,
    write_analysis_result,
)

Predicate = Callable[[Prediction], bool]

LOW_CONFIDENCE = 0.70
MODERATE_CONFIDENCE = 0.90
_ILLUSTRATIVE_SAMPLE = (13, 15)
_EXPECTED_ACCURACY = 0.87
_HALF_WIDTHS = (0.10, 0.05)


def _any_internal_signal(pred: Prediction) -> bool:
    return (
        pred["confidence"] < LOW_CONFIDENCE
        or has_structural_inconsistency(pred)
        or has_inter_agent_conflict(pred)
    )


SIGNALS: dict[str, Predicate] = {
    f"confidence < {LOW_CONFIDENCE:.2f}": lambda p: p["confidence"] < LOW_CONFIDENCE,
    f"confidence < {MODERATE_CONFIDENCE:.2f}": lambda p: p["confidence"] < MODERATE_CONFIDENCE,
    "structural inconsistency": has_structural_inconsistency,
    "cross_check conflict": has_inter_agent_conflict,
    "any internal signal": _any_internal_signal,
}


def signal_report(items: list[Prediction], is_flagged: Predicate) -> dict[str, float | int | None]:
    flagged = [p for p in items if is_flagged(p)]
    total_errors = sum(map(is_type_error, items))
    flagged_errors = sum(map(is_type_error, flagged))
    unflagged_count = len(items) - len(flagged)
    return {
        "n": len(items),
        "flagged": len(flagged),
        "flagged_share": ratio(len(flagged), len(items)),
        "error_rate_flagged": ratio(flagged_errors, len(flagged)),
        "error_rate_unflagged": ratio(total_errors - flagged_errors, unflagged_count),
        "base_error_rate": ratio(total_errors, len(items)),
        "errors_captured": ratio(flagged_errors, total_errors),
    }


def index_by_text(run: RunData) -> dict[str, Prediction]:
    return {p["text"]: p for p in run.predictions if not is_parse_failure(p)}


def disagreement_report(
    per_model: dict[str, dict[str, Prediction]], reference: str, texts: list[str]
) -> dict[str, float | int | str | None]:
    others = [m for m in per_model if m != reference]
    disagree: list[Prediction] = []
    unanimous: list[Prediction] = []
    for text in texts:
        reference_pred = per_model[reference][text]
        differs = any(
            per_model[o][text]["requirement_type"] != reference_pred["requirement_type"]
            for o in others
        )
        (disagree if differs else unanimous).append(reference_pred)
    total_errors = sum(map(is_type_error, disagree + unanimous))
    disagree_errors = sum(map(is_type_error, disagree))
    return {
        "reference": reference,
        "flagged": len(disagree),
        "flagged_share": ratio(len(disagree), len(texts)),
        "error_rate_disagree": ratio(disagree_errors, len(disagree)),
        "error_rate_unanimous": ratio(total_errors - disagree_errors, len(unanimous)),
        "errors_captured": ratio(disagree_errors, total_errors),
    }


def majority_vote_accuracy(per_model: dict[str, dict[str, Prediction]], texts: list[str]) -> float:
    """Ties resolve to the vote of the first model in sorted order (deterministic)."""
    first_model = next(iter(per_model.values()))
    correct = 0
    for text in texts:
        votes = Counter(per_model[m][text]["requirement_type"] for m in per_model)
        winner = votes.most_common(1)[0][0]
        correct += winner == first_model[text]["metadata"]["label_type"]
    return correct / len(texts)


def accuracy_interval(run: RunData) -> dict[str, float | int | str | None]:
    items = [p for p in run.predictions if not is_parse_failure(p)]
    correct = sum(not is_type_error(p) for p in items)
    low, high = wilson_interval(correct, len(items))
    return {
        "model": run.model,
        "lang": run.lang,
        "n": len(items),
        "accuracy": ratio(correct, len(items)),
        "ci95_low": low,
        "ci95_high": high,
    }


def model_disagreement_section(runs: list[RunData]) -> dict[str, dict[str, Any]]:
    section: dict[str, dict[str, Any]] = {}
    for lang, models in latest_run_per_model_and_lang(runs).items():
        if len(models) < 2:
            continue
        per_model = {name: index_by_text(run) for name, run in models.items()}
        texts = sorted(set.intersection(*(set(index) for index in per_model.values())))
        section[lang] = {
            "models": list(per_model),
            "common_items": len(texts),
            "reference_reports": [disagreement_report(per_model, m, texts) for m in per_model],
            "majority_vote_accuracy": majority_vote_accuracy(per_model, texts),
        }
    return section


def _print_signals(report: dict[str, dict[str, float | int | None]]) -> None:
    print(
        f"{'signal':<28}{'flagged':>9}{'share':>8}{'err|flag':>10}{'err|rest':>10}{'captured':>10}"
    )
    for name, r in report.items():
        print(
            f"{name:<28}{r['flagged']:>9}{format_percent(r['flagged_share']):>8}"
            f"{format_percent(r['error_rate_flagged']):>10}"
            f"{format_percent(r['error_rate_unflagged']):>10}"
            f"{format_percent(r['errors_captured']):>10}"
        )


def _print_disagreement(section: dict[str, dict[str, Any]]) -> None:
    for lang, block in section.items():
        print(f"\nlang={lang} models={block['models']} common items={block['common_items']}")
        for r in block["reference_reports"]:
            print(
                f"  ref={r['reference']:<24} flagged={format_percent(r['flagged_share']):>6} "
                f"err|disagree={format_percent(r['error_rate_disagree']):>6} "
                f"err|unanimous={format_percent(r['error_rate_unanimous']):>6} "
                f"captured={format_percent(r['errors_captured']):>6}"
            )
        print(f"  majority-vote accuracy: {block['majority_vote_accuracy']:.3f}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", default="pipeline")
    parser.add_argument("--n", type=int, default=FULL_DATASET_N)
    parser.add_argument("--generation", choices=list(GENERATIONS), default=HISTORICAL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    runs = load_full_runs(strategy=args.strategy, n=args.n, generation=args.generation)
    items = [p for run in runs for p in run.predictions if not is_parse_failure(p)]

    signals = {name: signal_report(items, pred) for name, pred in SIGNALS.items()}
    disagreement = model_disagreement_section(runs)
    intervals = [accuracy_interval(run) for run in runs]
    successes, total = _ILLUSTRATIVE_SAMPLE
    illustrative_low, illustrative_high = wilson_interval(successes, total)
    sample_sizes = {
        f"{half_width:.2f}": required_sample_size(_EXPECTED_ACCURACY, half_width)
        for half_width in _HALF_WIDTHS
    }

    print(f"runs={len(runs)} items (without parse failures)={len(items)}\n")
    _print_signals(signals)
    _print_disagreement(disagreement)
    print(
        f"\n{successes}/{total} -> Wilson 95% [{illustrative_low:.3f}, {illustrative_high:.3f}]; "
        f"n needed for a half-width of +-0.10 / +-0.05: {list(sample_sizes.values())}"
    )

    manifest = build_analysis_manifest(
        "analyze_error_signals",
        InformationRegime.ZERO_SHOT,
        {
            "strategy": args.strategy,
            "n": args.n,
            "generation": args.generation,
            "runs_loaded": len(runs),
        },
    )
    results = {
        "n_items": len(items),
        "signals": signals,
        "model_disagreement": disagreement,
        "accuracy_intervals": intervals,
        "sample_size_for_half_width": sample_sizes,
    }
    print(f"\nwritten: {write_analysis_result('error_signals', manifest, results)}")


if __name__ == "__main__":
    main()
