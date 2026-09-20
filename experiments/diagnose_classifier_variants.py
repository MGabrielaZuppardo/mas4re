"""Which agentic component moves the classifier's accuracy? (uses the LLM)

Runs one sample through variants of ClassificationAgent that differ in a single component:
  pre_adr011       one call, no memory, no tool, no critique (behaviour before ADR-011)
  current          production agent (memory + tool + gated critique)
  no_critique      current, critique disabled
  no_memory        current, episodic memory disabled
  always_critique  current, critique on every item. Memory retention is gated on the critique
                   NOT firing, so this variant also retains nothing.

Usage (from the repository root, Ollama running; keep other Ollama work idle):
    python -m experiments.diagnose_classifier_variants
    python -m experiments.diagnose_classifier_variants --n 15 --variants current no_critique

Output:
    experiments/results/analysis/classifier_variants_<timestamp>.json
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from agents.base import llm_retry
from agents.classifier import ClassificationAgent
from domain.models import (
    ClassificationOutput,
    ClassifiedRequirement,
    Requirement,
)
from experiments.analysis_common import (
    DEFAULT_SEED,
    PARSE_FAILURE_PREFIX,
    ZERO_SHOT,
    build_analysis_manifest,
    load_promise_sample,
    write_analysis_result,
)
from prompts.v1.classification import build_classification_messages

DEFAULT_MODEL = "ollama/qwen2.5:7b"
DEFAULT_SAMPLE_SIZE = 15


class _CritiqueCounter:
    """Mixin: counts critique calls and which way they change the classification."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stats: Counter[str] = Counter()

    def _critique(
        self, requirement_text: str, output: ClassificationOutput
    ) -> ClassificationOutput:
        revised = super()._critique(requirement_text, output)  # type: ignore[misc]
        self.stats["critique_calls"] += 1
        if revised.requirement_type != output.requirement_type:
            self.stats[
                f"flip_{output.requirement_type.value}_to_{revised.requirement_type.value}"
            ] += 1
        elif revised.nfr_category != output.nfr_category:
            self.stats["category_change"] += 1
        return revised


class CurrentClassifier(_CritiqueCounter, ClassificationAgent):
    pass


class NoCritiqueClassifier(_CritiqueCounter, ClassificationAgent):
    def _needs_critique(self, output: ClassificationOutput) -> bool:
        return False


class AlwaysCritiqueClassifier(_CritiqueCounter, ClassificationAgent):
    def _needs_critique(self, output: ClassificationOutput) -> bool:
        return True


class _WithoutMemory:
    def classify_batch(
        self, requirements: list[Requirement], max_workers: int | None = None
    ) -> list[ClassifiedRequirement]:
        self._memory = None  # type: ignore[attr-defined]
        return self._run_batch(  # type: ignore[attr-defined]
            requirements, stage="classify", max_workers=max_workers
        )


class NoMemoryClassifier(_WithoutMemory, _CritiqueCounter, ClassificationAgent):
    pass


class PreAdr011Classifier(_WithoutMemory, ClassificationAgent):
    @llm_retry
    def _process_single(self, requirement: Requirement) -> ClassifiedRequirement:
        messages = build_classification_messages(
            requirement_text=requirement.text,
            lang=self._lang,
            nfr_categories=self._nfr_categories,
        )
        output = self._parse_response(str(self._llm.invoke(messages).content), requirement.id)
        return ClassifiedRequirement.from_requirement(requirement, output)


VARIANTS: dict[str, type[ClassificationAgent]] = {
    "pre_adr011": PreAdr011Classifier,
    "current": CurrentClassifier,
    "no_critique": NoCritiqueClassifier,
    "no_memory": NoMemoryClassifier,
    "always_critique": AlwaysCritiqueClassifier,
}


def evaluate(agent: ClassificationAgent, sample: list[Requirement]) -> dict[str, Any]:
    started = time.monotonic()
    classified = agent.classify_batch(sample)
    elapsed = time.monotonic() - started

    by_id = {c.id: c for c in classified}
    truth = [r.metadata["label_type"] for r in sample]
    predictions = [by_id[r.id].requirement_type.value if r.id in by_id else "?" for r in sample]
    confidences = [c.confidence for c in classified]
    return {
        "classified": len(classified),
        "accuracy": accuracy_score(truth, predictions),
        "f1_macro": f1_score(truth, predictions, average="macro", zero_division=0),
        "predicted_f": predictions.count("F"),
        "predicted_nf": predictions.count("NF"),
        "parse_failures": sum(c.justification.startswith(PARSE_FAILURE_PREFIX) for c in classified),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "elapsed_seconds": round(elapsed, 1),
        "critique": dict(getattr(agent, "stats", {})),
        "truth": truth,
        "predictions": predictions,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    sample = load_promise_sample(args.n, args.seed)
    truth_f = sum(r.metadata["label_type"] == "F" for r in sample)
    print(f"sample: {truth_f} F / {len(sample) - truth_f} NF | model={args.model}\n")

    results: dict[str, dict[str, Any]] = {}
    for name in args.variants:
        results[name] = evaluate(VARIANTS[name](model=args.model), sample)
        r = results[name]
        print(
            f"{name:<16} n={r['classified']}/{len(sample)} acc={r['accuracy']:.3f} "
            f"f1={r['f1_macro']:.3f} predF={r['predicted_f']} predNF={r['predicted_nf']} "
            f"parse_fail={r['parse_failures']} t={r['elapsed_seconds']:.0f}s "
            f"critique={r['critique'] or '-'}",
            flush=True,
        )

    manifest = build_analysis_manifest(
        "diagnose_classifier_variants",
        ZERO_SHOT,
        {"model": args.model, "n": args.n, "seed": args.seed, "variants": args.variants},
    )
    print(f"\nwritten: {write_analysis_result('classifier_variants', manifest, results)}")


if __name__ == "__main__":
    main()
