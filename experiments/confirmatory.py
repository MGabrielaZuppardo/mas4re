"""The frozen confirmatory design and the check that a run follows it (ADR-014)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

SPEC_PATH = Path(__file__).parent / "configs" / "confirmatory.yml"

_FROZEN_FIELDS = (
    "dataset_name",
    "data_lang",
    "lang",
    "seed",
    "temperature",
    "model",
    "information_regime",
)


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def spec_sha256(path: Path = SPEC_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ARMS = ("p0", "p0-repeat", "p2", "p3", "control")


def arm_run_values(arm: str, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every value of a `run` for one arm of the frozen design, read from the spec: changing the
    model, language, seed or dataset of the experiment means changing the spec and nothing else."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)}, got {arm!r}")
    spec = spec or load_spec()
    noise = spec["noise_floor"]
    if arm == "control":
        strategy = spec["control"]
    else:
        level = noise["protocol"] if arm == "p0-repeat" else arm.upper()
        strategy = spec["protocols"][level]
    classifier, _, prioritizer = spec["model"].partition("+")
    return {
        "strategy": strategy,
        "model": classifier,
        "pri_model": prioritizer or classifier,
        "lang": spec["lang"],
        "data_lang": spec["data_lang"],
        "dataset_name": spec["dataset_name"],
        "seed": spec["seed"],
        "temperature": spec["temperature"],
        "max_steps": spec["max_steps"],
        "n": spec["n_samples"],
        "tag": noise["tag"] if arm == "p0-repeat" else None,
    }


def assess(
    strategy_name: str,
    config_values: dict[str, Any],
    n_samples: int | None,
    n_loaded: int,
    spec: dict[str, Any] | None = None,
    tag: str | None = None,
    max_steps: int | None = None,
    exploratory_variant: str | None = None,
) -> dict[str, Any]:
    """Whether a run is part of the confirmatory design, and every way it deviates from it."""
    spec = spec or load_spec()
    deviations = [
        f"{name}: {config_values.get(name)!r} != {spec[name]!r}"
        for name in _FROZEN_FIELDS
        if config_values.get(name) != spec[name]
    ]
    if exploratory_variant is not None:
        deviations.append(f"exploratory variant: {exploratory_variant}")
    allowed = {*spec["protocols"].values(), spec["control"]}
    if strategy_name not in allowed:
        deviations.append(f"strategy: {strategy_name!r} not in {sorted(allowed)}")
    if strategy_name in spec["protocols"].values() and max_steps != spec["max_steps"]:
        deviations.append(f"max_steps: {max_steps!r} != {spec['max_steps']!r}")
    noise_tag = spec["noise_floor"]["tag"]
    if tag is not None and tag != noise_tag:
        deviations.append(f"tag: {tag!r} is not the noise-floor label {noise_tag!r}")
    elif tag == noise_tag and strategy_name != spec["protocols"][spec["noise_floor"]["protocol"]]:
        deviations.append(
            f"tag: {tag!r} only applies to the {spec['noise_floor']['protocol']} strategy"
        )
    if n_samples != spec["n_samples"]:
        deviations.append(f"n_samples: {n_samples!r} != {spec['n_samples']!r}")
    if n_loaded != spec["expected_n"]:
        deviations.append(f"dataset_n: {n_loaded} != {spec['expected_n']}")
    return {
        "is_confirmatory": not deviations,
        "spec_sha256": spec_sha256(),
        "deviations": deviations,
    }
