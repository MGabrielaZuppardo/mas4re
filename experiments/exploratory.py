"""Exploratory variants of the experiment (ADR-016), kept apart from the confirmatory design.

A variant starts from the frozen confirmatory values and changes only what it declares: the model,
the mediator's Must limit, or the warm-up the prioritizer's memory starts from. Nothing here edits
`confirmatory.yml`, and a run made from a variant can never be taken for a confirmatory one.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from agents.memory import MemoryItem, WarmupSeed
from domain.agent_actions import EpisodePhase
from domain.models import Requirement
from experiments.confirmatory import ARMS, arm_run_values, load_spec

SPEC_PATH = Path(__file__).parent / "configs" / "exploratory.yml"
EXPLORATORY_DIR = Path("experiments") / "exploratory"
RESULTS_DIR = EXPLORATORY_DIR / "results"
TRACES_DIR = EXPLORATORY_DIR / "traces"
ANALYSIS_DIR = EXPLORATORY_DIR / "analysis"


def load_exploratory_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def exploratory_sha256(path: Path = SPEC_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_sha256(variant: str, spec: dict[str, Any] | None = None) -> str:
    """Hash of one variant's declaration, so that adding or editing another variant leaves it as
    it was."""
    canonical = json.dumps(variant_spec(variant, spec), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def variant_spec(variant: str, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or load_exploratory_spec()
    if variant not in spec["variants"]:
        raise ValueError(f"variant must be one of {sorted(spec['variants'])}, got {variant!r}")
    return dict(spec["variants"][variant])


def variant_run_values(
    variant: str,
    arm: str,
    confirmatory: dict[str, Any] | None = None,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The values of one run of a variant: the confirmatory arm, with only the declared changes."""
    declared = variant_spec(variant, spec)
    if arm not in ARMS or arm not in declared["arms"]:
        raise ValueError(f"arm must be one of {declared['arms']} for {variant}, got {arm!r}")
    base = dict(confirmatory or load_spec())
    if declared.get("model"):
        base["model"] = declared["model"]
    values = arm_run_values(arm, base)
    values["must_limit"] = declared.get("must_limit")
    values["pin_warmup_from"] = declared.get("pin_warmup_from")
    values["tag"] = variant if arm != "p0-repeat" else f"{variant}-repeat"
    return values


def manifest_block(variant: str, arm: str, pinned_run: str | None) -> dict[str, Any]:
    declared = variant_spec(variant)
    return {
        "variant": variant,
        "arm": arm,
        "spec_sha256": variant_sha256(variant),
        "question": declared["question"],
        "hypothesis": " ".join(str(declared["hypothesis"]).split()),
        "must_limit": declared.get("must_limit"),
        "warmup_pinned_from": pinned_run,
    }


def warmup_seed_from_run(
    run_dir: Path, requirements: Sequence[Requirement], size: int
) -> WarmupSeed:
    """The first `size` prioritizer decisions of a finished run, as the memory they would have
    left: the requirement's text, the priority decided the first time, and the justification."""
    texts = {r.id: r.text for r in requirements}
    initial = {
        row["id"]: row["initial_priority"]
        for row in map(json.loads, (run_dir / "decisions.jsonl").read_text("utf-8").splitlines())
    }
    items: list[MemoryItem] = []
    for line in (run_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines():
        episode = json.loads(line)
        if episode["agent"] != "prioritizer" or episode["phase"] != EpisodePhase.WORK.value:
            continue
        submit = next((a for a in episode["actions"] if a["name"] == "submit_priority"), None)
        requirement_id = episode["requirement_id"]
        if submit is None or requirement_id not in texts or initial.get(requirement_id) is None:
            continue
        items.append(
            MemoryItem(
                requirement_id,
                texts[requirement_id],
                output_summary=initial[requirement_id],
                justification=submit["arguments"].get("justification", ""),
            )
        )
        if len(items) == size:
            break
    decided = Counter(item.output_summary for item in items)
    return WarmupSeed(items=tuple(items), decided=tuple(sorted(decided.items())))
