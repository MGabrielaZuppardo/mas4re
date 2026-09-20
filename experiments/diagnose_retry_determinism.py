"""Does a blind retry reproduce the first pass? (uses the LLM)

Condition B's retry resends the same batch with the same prompts, a fresh memory and
temperature 0.0, and tells the agents nothing about the conflict. If the outputs of a second
pass equal the first, extra rounds add cost but no information. This script processes the same
batch twice through the production agents and lists every item whose output changed.

Usage (from the repository root, Ollama running; keep other Ollama work idle):
    python -m experiments.diagnose_retry_determinism
    python -m experiments.diagnose_retry_determinism --n 10

Output:
    experiments/results/analysis/retry_determinism_<timestamp>.json
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from domain.models import ClassifiedRequirement, PrioritizedRequirement
from experiments.analysis_common import (
    DEFAULT_SEED,
    ZERO_SHOT,
    build_analysis_manifest,
    load_promise_sample,
    write_analysis_result,
)

DEFAULT_CLASSIFIER_MODEL = "ollama/qwen2.5:7b"
DEFAULT_PRIORITIZER_MODEL = "ollama/llama3.1:8b"
DEFAULT_SAMPLE_SIZE = 15

Signature = tuple[Any, ...]


def classification_signatures(items: list[ClassifiedRequirement]) -> dict[str, Signature]:
    return {c.id: (c.requirement_type.value, c.nfr_category, c.confidence) for c in items}


def priority_signatures(items: list[PrioritizedRequirement]) -> dict[str, Signature]:
    return {p.id: (p.priority.value if p.priority else None, p.priority_score) for p in items}


def differences(first: dict[str, Signature], second: dict[str, Signature]) -> list[dict[str, Any]]:
    return [
        {"id": item_id, "first": list(first[item_id]), "second": list(second[item_id])}
        for item_id in first
        if item_id in second and first[item_id] != second[item_id]
    ]


def compare_passes(label: str, first: dict[str, Signature], second: dict[str, Signature]) -> dict:
    changed = differences(first, second)
    print(
        f"{label}: {len(changed)}/{len(first)} items differ between pass 1 and pass 2", flush=True
    )
    for entry in changed:
        print(f"    {entry['id']}: {entry['first']} -> {entry['second']}", flush=True)
    return {"items": len(first), "differing": len(changed), "differences": changed}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clf-model", default=DEFAULT_CLASSIFIER_MODEL)
    parser.add_argument("--pri-model", default=DEFAULT_PRIORITIZER_MODEL)
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    sample = load_promise_sample(args.n, args.seed)

    started = time.monotonic()
    classifier = ClassificationAgent(model=args.clf_model)
    first_pass = classifier.classify_batch(sample)
    second_pass = classifier.classify_batch(sample)
    classification = compare_passes(
        "classifier ", classification_signatures(first_pass), classification_signatures(second_pass)
    )

    prioritizer = PrioritizationAgent(model=args.pri_model)
    first_priorities = prioritizer.prioritize_batch(first_pass)
    second_priorities = prioritizer.prioritize_batch(first_pass)
    prioritization = compare_passes(
        "prioritizer", priority_signatures(first_priorities), priority_signatures(second_priorities)
    )
    print(f"\nelapsed: {time.monotonic() - started:.0f}s")

    manifest = build_analysis_manifest(
        "diagnose_retry_determinism",
        ZERO_SHOT,
        {
            "classifier_model": args.clf_model,
            "prioritizer_model": args.pri_model,
            "n": args.n,
            "seed": args.seed,
        },
    )
    results = {"classification": classification, "prioritization": prioritization}
    print(f"written: {write_analysis_result('retry_determinism', manifest, results)}")


if __name__ == "__main__":
    main()
