"""Exploratory analyses (ADR-016). Descriptive, never confirmatory.

Two analyses, both labelled exploratory in their output:

  --warmup           X0: no new runs. How the first items decided in each confirmatory run relate
                     to the run's final Must share, and the protocol contrasts taking the run that
                     fell in the high-Must regime (the P0 repeat) as the baseline.
  --variant ID       The runs of one exploratory variant, against the confirmatory runs that share
                     its model and items. A variant on another model is read in its own cell only.

Effect sizes come with a bootstrap interval and sit beside the noise floor; no verdict is drawn from
a p-value, and nothing here is corrected for multiplicity.

Usage (from the repository root):
    python -m experiments.analyze_exploratory --warmup
    python -m experiments.analyze_exploratory --variant x2-mediator-active
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from domain.agent_actions import EpisodePhase
from experiments.analysis_common import build_analysis_manifest, write_analysis_result
from experiments.analyze_agency import AgencyRun, load_run
from experiments.analyze_protocols import (
    cost_report,
    find_arms,
    interaction_effect,
    quality_contrasts,
)
from experiments.architecture_claim import evaluate_arm
from experiments.confirmatory import load_spec
from experiments.exploratory import (
    ANALYSIS_DIR,
    RESULTS_DIR,
    load_exploratory_spec,
    variant_spec,
)

WARNING = (
    "EXPLORATORY: descriptive analysis outside the confirmatory design. One run per arm, one "
    "model, labels produced by the benchmark. Do not report these figures as confirmatory evidence."
)
CONFIRMATORY_RESULTS = Path("experiments") / "results"
_ARM_LABELS = {"p0": "P0", "p0-repeat": "P0R", "p2": "P2", "p3": "P3"}
_N_PERMUTATIONS = 2_000
_N_BOOTSTRAP = 2_000
_SEED = 42
_WITHIN_CELL = (("P2", "P0"), ("P3", "P0"), ("P3", "P2"), ("P0R", "P0"))


def warmup_signature(run: AgencyRun, size: int) -> list[str | None]:
    """First-pass priority of the first `size` items the prioritizer decided, in the order it
    decided them (the warm-up runs one item at a time)."""
    initial = {row["id"]: row["initial_priority"] for row in run.decisions}
    order = [
        e.requirement_id
        for e in run.episodes
        if e.agent == "prioritizer" and e.phase is EpisodePhase.WORK
    ]
    return [initial.get(requirement_id) for requirement_id in order[:size]]


def _must_share(run: AgencyRun) -> dict[str, Any]:
    musts = sum(1 for item in run.items.values() if item.priority and item.priority.value == "M")
    return {"musts": musts, "share": round(musts / len(run.items), 4) if run.items else None}


def warmup_table(runs: dict[str, AgencyRun], size: int) -> list[dict[str, Any]]:
    rows = [
        {
            "run": label,
            "warmup": "".join(s or "?" for s in warmup_signature(run, size)),
            **_must_share(run),
        }
        for label, run in runs.items()
    ]
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row["warmup"], []).append(row["run"])
    for row in rows:
        row["same_warmup_as"] = [r for r in groups[row["warmup"]] if r != row["run"]]
    return rows


def _descriptive(report: dict[str, Any]) -> dict[str, Any]:
    """Keep effect sizes and intervals; drop the p-values, which would invite a verdict."""
    for metrics in report["tasks"].values():
        for contrasts in metrics.values():
            for row in contrasts.values():
                if isinstance(row, dict):
                    row.pop("p_permutation", None)
                    row.pop("p_holm", None)
    return report


def contrasts_against(
    runs: dict[str, AgencyRun], pairs: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    return _descriptive(
        quality_contrasts(runs, _N_PERMUTATIONS, _N_BOOTSTRAP, _SEED, contrasts=pairs)
    )


def load_variant_runs(variant: str, results_dir: Path = RESULTS_DIR) -> dict[str, AgencyRun]:
    found: dict[str, list[Path]] = {}
    for manifest_path in sorted(results_dir.glob("*/manifest.json")):
        block = json.loads(manifest_path.read_text(encoding="utf-8")).get("exploratory") or {}
        label = _ARM_LABELS.get(block.get("arm", ""))
        if block.get("variant") == variant and label:
            found.setdefault(label, []).append(manifest_path.parent)
    duplicated = {k: [p.name for p in paths] for k, paths in found.items() if len(paths) > 1}
    if duplicated:
        raise ValueError(f"more than one run of the same arm of {variant}: {duplicated}")
    if not found:
        raise ValueError(f"no run of {variant} in {results_dir}")
    return {label: load_run(paths[0]) for label, paths in found.items()}


def _require_same_model(runs: dict[str, AgencyRun]) -> None:
    models = {run.manifest["model"] for run in runs.values()}
    if len(models) != 1:
        raise ValueError(f"refusing to compare runs of different models: {sorted(models)}")


def _present(pairs: Sequence[tuple[str, str]], runs: dict[str, AgencyRun]) -> list[tuple[str, str]]:
    return [(a, b) for a, b in pairs if a in runs and b in runs]


def variant_report(variant: str) -> dict[str, Any]:
    declared = variant_spec(variant)
    runs = load_variant_runs(variant)
    size = load_exploratory_spec()["warmup_size"]
    limits = load_spec()["architecture_claim"]
    protocols = {k: v for k, v in runs.items() if k in ("P0", "P2", "P3")}
    report: dict[str, Any] = {
        "variant": variant,
        "question": declared["question"],
        "hypothesis": " ".join(str(declared["hypothesis"]).split()),
        "runs": {label: run.path.name for label, run in runs.items()},
        "architecture": {
            label: evaluate_arm(label, run, limits) for label, run in protocols.items()
        },
        "interaction_effect": interaction_effect(runs),
        "cost": cost_report(protocols) if len(protocols) == 3 else None,
    }
    if declared.get("model"):
        _require_same_model(runs)
        report["warmup"] = warmup_table(runs, size)
        report["contrasts"] = contrasts_against(runs, _present(_WITHIN_CELL, runs))
        return report
    references = {
        f"R:{label}": load_run(path) for label, path in find_arms(CONFIRMATORY_RESULTS).items()
    }
    combined = {**{f"V:{k}": v for k, v in runs.items()}, **references}
    _require_same_model(combined)
    report["warmup"] = warmup_table(combined, size)
    wanted = [(f"V:{v}", f"R:{r}") for v in sorted(runs) for r in ("P0", "P0R", "P3")]
    report["contrasts"] = contrasts_against(combined, _present(wanted, combined))
    return report


def warmup_report() -> dict[str, Any]:
    runs = {label: load_run(path) for label, path in find_arms(CONFIRMATORY_RESULTS).items()}
    size = load_exploratory_spec()["warmup_size"]
    return {
        "question": "Does the warm-up decide the Must regime of a run?",
        "warmup": warmup_table(runs, size),
        "sensitivity_baseline_P0R": contrasts_against(
            runs, [("P0", "P0R"), ("P2", "P0R"), ("P3", "P0R")]
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--warmup", action="store_true")
    group.add_argument("--variant")
    args = parser.parse_args(argv)
    name = "warmup" if args.warmup else args.variant
    body = warmup_report() if args.warmup else variant_report(args.variant)
    reference = load_run(next(iter(find_arms(CONFIRMATORY_RESULTS).values())))
    manifest = build_analysis_manifest(
        "analyze_exploratory",
        reference.information_regime,
        {"analysis": name},
        dataset_path=str(reference.manifest["dataset"]),
    )
    result = {"status": "exploratory", "warning": WARNING, **body}
    path = write_analysis_result(f"exploratory_{name}", manifest, result, ANALYSIS_DIR)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
