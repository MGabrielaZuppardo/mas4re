"""Which agentic component drives Must-Have inflation in the prioritizer? (uses the LLM)

The sample is classified ONCE by the production classifier, so every prioritizer variant
receives exactly the same input. Only the prioritizer changes:
  pre_adr011   one call, no memory, no critique (behaviour before ADR-011)
  current      production agent (memory + critique on every item)
  no_critique  current, critique disabled
  no_memory    current, episodic memory disabled

Usage (from the repository root, Ollama running; keep other Ollama work idle):
    python -m experiments.diagnose_prioritizer_variants
    python -m experiments.diagnose_prioritizer_variants --n 15 --variants pre_adr011 current

Output:
    experiments/results/analysis/prioritizer_variants_<timestamp>.json
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from typing import Any

from agents.base import llm_retry, rank_by_priority
from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from domain.models import (
    ClassifiedRequirement,
    PrioritizationOutput,
    PrioritizedRequirement,
    Requirement,
)
from evaluation.cross_agent_check import check_requirements
from evaluation.metrics.prioritization import compute_moscow_distribution
from experiments.analysis_common import (
    DEFAULT_SEED,
    ZERO_SHOT,
    build_analysis_manifest,
    load_promise_sample,
    write_analysis_result,
)
from experiments.analyze_coordination_power import DSDM_MUST_LIMIT, INTEGRATION_TEST_MUST_LIMIT
from prompts.v1.prioritization import build_prioritization_messages

DEFAULT_CLASSIFIER_MODEL = "ollama/qwen2.5:7b"
DEFAULT_PRIORITIZER_MODEL = "ollama/llama3.1:8b"
DEFAULT_SAMPLE_SIZE = 15


class _PriorityFlipCounter:
    """Mixin: counts critique calls and which way they change the priority."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stats: Counter[str] = Counter()

    def _critique(
        self, requirement_text: str, output: PrioritizationOutput
    ) -> PrioritizationOutput:
        revised = super()._critique(requirement_text, output)  # type: ignore[misc]
        self.stats["critique_calls"] += 1
        if revised.priority != output.priority:
            self.stats[f"flip_{output.priority.value}_to_{revised.priority.value}"] += 1
        return revised


class CurrentPrioritizer(_PriorityFlipCounter, PrioritizationAgent):
    pass


class NoCritiquePrioritizer(PrioritizationAgent):
    def _critique(
        self, requirement_text: str, output: PrioritizationOutput
    ) -> PrioritizationOutput:
        return output


class _WithoutMemory:
    def prioritize_batch(
        self, requirements: list[ClassifiedRequirement], max_workers: int | None = None
    ) -> list[PrioritizedRequirement]:
        self._memory = None  # type: ignore[attr-defined]
        ordered = self._run_batch(  # type: ignore[attr-defined]
            requirements, stage="prioritize", max_workers=max_workers
        )
        return rank_by_priority(ordered)


class NoMemoryPrioritizer(_WithoutMemory, _PriorityFlipCounter, PrioritizationAgent):
    pass


class PreAdr011Prioritizer(_WithoutMemory, PrioritizationAgent):
    @llm_retry
    def _process_single(self, requirement: ClassifiedRequirement) -> PrioritizedRequirement:
        messages = build_prioritization_messages(
            requirement_text=requirement.text,
            requirement_type=requirement.requirement_type.value,
            nfr_category=requirement.nfr_category,
            lang=self._lang,
        )
        output = self._parse_response(str(self._llm.invoke(messages).content), requirement.id)
        return PrioritizedRequirement.from_classified(requirement, output)


VARIANTS: dict[str, type[PrioritizationAgent]] = {
    "pre_adr011": PreAdr011Prioritizer,
    "current": CurrentPrioritizer,
    "no_critique": NoCritiquePrioritizer,
    "no_memory": NoMemoryPrioritizer,
}


def evaluate(agent: PrioritizationAgent, classified: list[ClassifiedRequirement]) -> dict[str, Any]:
    started = time.monotonic()
    prioritized = agent.prioritize_batch(classified)
    elapsed = time.monotonic() - started

    distribution = compute_moscow_distribution(prioritized)
    must = distribution.get("M", 0.0)
    by_id = {p.id: p for p in prioritized}
    return {
        "prioritized": len(prioritized),
        "distribution": distribution,
        "must_share": must,
        f"over_{INTEGRATION_TEST_MUST_LIMIT:.2f}": must > INTEGRATION_TEST_MUST_LIMIT,
        f"over_{DSDM_MUST_LIMIT:.2f}": must > DSDM_MUST_LIMIT,
        "cross_check_conflicts": len(check_requirements(prioritized)),
        "elapsed_seconds": round(elapsed, 1),
        "critique": dict(getattr(agent, "stats", {})),
        "category_and_priority": [
            f"{by_id[c.id].nfr_category or 'F'}:{by_id[c.id].priority.value}"  # type: ignore[union-attr]
            for c in classified
            if c.id in by_id
        ],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clf-model", default=DEFAULT_CLASSIFIER_MODEL)
    parser.add_argument("--pri-model", default=DEFAULT_PRIORITIZER_MODEL)
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    sample: list[Requirement] = load_promise_sample(args.n, args.seed)

    started = time.monotonic()
    classified = ClassificationAgent(model=args.clf_model).classify_batch(sample)
    categories = Counter(c.nfr_category or "F" for c in classified)
    print(
        f"input: {len(classified)}/{len(sample)} classified in {time.monotonic() - started:.0f}s "
        f"| {dict(categories)}\n",
        flush=True,
    )

    results: dict[str, dict[str, Any]] = {}
    for name in args.variants:
        results[name] = evaluate(VARIANTS[name](model=args.pri_model), classified)
        r = results[name]
        print(
            f"{name:<12} n={r['prioritized']}/{len(classified)} dist={r['distribution']} "
            f"must={r['must_share']:.0%} conflicts={r['cross_check_conflicts']} "
            f"t={r['elapsed_seconds']:.0f}s critique={r['critique'] or '-'}",
            flush=True,
        )
        print("    ", " ".join(r["category_and_priority"]), flush=True)

    manifest = build_analysis_manifest(
        "diagnose_prioritizer_variants",
        ZERO_SHOT,
        {
            "classifier_model": args.clf_model,
            "prioritizer_model": args.pri_model,
            "n": args.n,
            "seed": args.seed,
            "variants": args.variants,
        },
    )
    print(f"\nwritten: {write_analysis_result('prioritizer_variants', manifest, results)}")


if __name__ == "__main__":
    main()
