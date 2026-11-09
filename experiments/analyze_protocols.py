"""Confirmatory analysis of the coordination protocols P0, P2 and P3 (ADR-014).

The factor is the coordination protocol, so the contrasts are P2-P0, P3-P0 and P3-P2 on the same
requirements, paired by id. The `pipeline` workflow is an external architectural control and is
not a level of the factor: it is only ever reported by `analyze_agency`.

Three families of outcomes are kept apart and never merged into one score:
  * agreement with the benchmark's reference labels (Macro-F1, MCC, kappa, accuracy), tested with a
    paired permutation test and a paired bootstrap CI, because they are corpus metrics with no
    per-requirement value;
  * operational reliability (completion, termination, failures per 100 calls);
  * cost per requirement (LLM calls, tokens, latency): Friedman across the protocols and Wilcoxon
    per contrast.
A quality-versus-cost table then says, for each contrast, whether one side dominates or whether
there is a trade-off.

Runs must come from the frozen design (`experiments/configs/confirmatory.yml`); anything else is
refused unless `--exploratory` is passed, and the output is then marked as such.

A second, identical P0 run is the noise floor: temperature 0 does not make repeated runs
bit-identical, so each protocol delta is reported beside the P0-versus-P0 delta of the same metric.

Usage (from the repository root):
    python -m experiments.analyze_protocols [--results-dir DIR]
        finds the confirmatory run of each arm (the repeat is the one run with --tag p0-repeat)
    python -m experiments.analyze_protocols --p0 RUN --p2 RUN --p3 RUN --p0-repeat RUN
        [--exploratory]    explicit paths take precedence over what was found

Output:
    experiments/results/analysis/protocols_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from agents.agentic.toolkit import REQUEST_RECLASSIFICATION
from domain.enums import MoSCoWPriority
from domain.models import PrioritizedRequirement
from evaluation.agent_activity import cost_by_phase, cost_per_requirement, message_flows
from evaluation.metrics import corpus
from evaluation.metrics.prioritization import ORDINAL_MAPPING_VERIFIED
from evaluation.metrics.statistical import (
    bootstrap_statistic_difference,
    friedman_test,
    holm_adjust,
    mcnemar_exact,
    paired_permutation_test,
    wilcoxon_test,
)
from experiments.analysis_common import build_analysis_manifest, write_analysis_result
from experiments.analyze_agency import AgencyRun, load_run, single_cell, single_regime
from experiments.architecture_claim import evaluate_arms
from experiments.confirmatory import load_spec, spec_sha256

PROTOCOLS = ("P0", "P2", "P3")
COSTS = ("llm_calls", "tokens", "latency_ms")
_MOSCOW_CLASSES = len(MoSCoWPriority)
_TYPE_CLASSES = 2
_COMPARABLE = ("model", "seed", "temperature", "dataset_md5", "dataset_n", "lang", "data_lang")

LIMITATIONS = (
    "One run per protocol plus one repeat of P0: the CIs describe the variation across "
    "requirements, not across repeated runs; the P0 repeat gives a single estimate of run-to-run "
    "noise, not its distribution, and temperature 0 does not guarantee identical repeated outputs.",
    "The benchmark labels are automatic reference criteria, not independent human truth: agreement "
    "means agreement with the benchmark.",
    "Census of one benchmark (PrioReSpa, Spanish, one GIS project) with one model: no claim about "
    "other requirement sets or other LLMs.",
    "MoSCoW is a priority categorisation, a subset of requirements prioritisation.",
)


class NotConfirmatoryError(ValueError):
    """Raised when a run is not part of the frozen confirmatory design."""


class UnpairedProtocolsError(ValueError):
    """Raised when the runs differ in anything but the coordination protocol."""


# ── loading and pairing ────────────────────────────────────────────────────────────────


def ensure_confirmatory(runs: dict[str, AgencyRun], exploratory: bool) -> None:
    if exploratory:
        return
    current = spec_sha256()
    for name, run in runs.items():
        status = run.manifest.get("confirmatory") or {}
        if not status.get("is_confirmatory"):
            reasons = status.get("deviations") or ["manifest has no confirmatory block"]
            raise NotConfirmatoryError(f"{name} ({run.path.name}) is exploratory: {reasons}")
        if status.get("spec_sha256") != current:
            raise NotConfirmatoryError(f"{name} ran under a different confirmatory spec")


def ensure_same_design(runs: dict[str, AgencyRun]) -> None:
    single_cell(list(runs.values()))
    single_regime(list(runs.values()))
    reference = next(iter(runs.values()))
    for name, run in runs.items():
        differing = [f for f in _COMPARABLE if run.manifest.get(f) != reference.manifest.get(f)]
        if differing:
            raise UnpairedProtocolsError(f"{name} differs from the others in {differing}")


# ── paired corpus metrics ──────────────────────────────────────────────────────────────


def _reference_type(item: PrioritizedRequirement) -> int:
    return int(item.metadata["label_type"] != "F")


def _predicted_type(item: PrioritizedRequirement) -> int:
    return int(item.requirement_type.value != "F")


def _reference_priority(item: PrioritizedRequirement) -> int:
    gold = item.reference_priority
    assert gold is not None
    return gold.ordinal - 1


def _predicted_priority(item: PrioritizedRequirement) -> int:
    assert item.priority is not None
    return item.priority.ordinal - 1


def _metrics() -> dict[str, dict[str, Callable[..., float]]]:
    def kappa_weighted(reference: np.ndarray, predicted: np.ndarray, k: int) -> float:
        return corpus.cohen_kappa(reference, predicted, k, weights="linear")

    moscow = {
        "f1_macro": corpus.f1_macro,
        "kappa_nominal": corpus.cohen_kappa,
        "weighted_kappa": kappa_weighted,
        "accuracy": corpus.accuracy,
    }
    return {
        "type": {
            "f1_macro": corpus.f1_macro,
            "mcc": corpus.matthews,
            "accuracy": corpus.accuracy,
        },
        "moscow": moscow,
    }


def _arrays(
    runs: dict[str, AgencyRun], ids: Sequence[str]
) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:
    sample = {i: next(iter(runs.values())).items[i] for i in ids}
    reference_type = np.array([_reference_type(sample[i]) for i in ids])
    out: dict[str, dict[str, Any]] = {"type": {"reference": reference_type, "predicted": {}}}
    out["moscow"] = {"reference": None, "predicted": {}}
    for name, run in runs.items():
        out["type"]["predicted"][name] = np.array([_predicted_type(run.items[i]) for i in ids])
        out["moscow"]["predicted"][name] = np.array(
            [_predicted_priority(run.items[i]) for i in ids]
        )
    if all(sample[i].reference_priority is not None for i in ids):
        out["moscow"]["reference"] = np.array([_reference_priority(sample[i]) for i in ids])
    return out


def _contrast(
    metric: Callable[..., float],
    k: int,
    reference: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    n = len(reference)

    def swapped(mask: np.ndarray) -> float:
        a, b = np.where(mask, second, first), np.where(mask, first, second)
        return metric(reference, a, k) - metric(reference, b, k)

    def resampled(index: np.ndarray) -> float:
        return metric(reference[index], first[index], k) - metric(
            reference[index], second[index], k
        )

    result: dict[str, Any] = {
        "first": round(float(metric(reference, first, k)), 4),
        "second": round(float(metric(reference, second, k)), 4),
    }
    try:
        test = paired_permutation_test(swapped, n, n_permutations, seed)
        interval = bootstrap_statistic_difference(resampled, n, n_bootstrap, seed=seed)
    except ValueError as error:
        return {**result, "verdict": f"undefined: {error}"}
    return {
        **result,
        "delta": test.observed_difference,
        "ci95": [interval.ci_lower, interval.ci_upper],
        "p_permutation": round(test.p_value, 6),
    }


PLANNED_CONTRASTS = (("P2", "P0"), ("P3", "P0"), ("P3", "P2"))
NOISE_CONTRAST = ("P0R", "P0")


def quality_contrasts(
    runs: dict[str, AgencyRun],
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
    contrasts: Sequence[tuple[str, str]] = PLANNED_CONTRASTS,
) -> dict[str, Any]:
    ids = sorted(set.intersection(*(set(run.items) for run in runs.values())))
    arrays = _arrays(runs, ids)
    metrics = _metrics()
    classes = {"type": _TYPE_CLASSES, "moscow": _MOSCOW_CLASSES}
    report: dict[str, Any] = {"n_paired": len(ids), "tasks": {}}
    for task, functions in metrics.items():
        reference = arrays[task]["reference"]
        if reference is None:
            continue
        predicted = arrays[task]["predicted"]
        block: dict[str, Any] = {}
        for metric_name, function in functions.items():
            rows = {
                f"{a}-{b}": _contrast(
                    function,
                    classes[task],
                    reference,
                    predicted[a],
                    predicted[b],
                    n_permutations,
                    n_bootstrap,
                    seed,
                )
                for a, b in contrasts
            }
            adjusted = holm_adjust(
                [row["p_permutation"] for row in rows.values() if "p_permutation" in row]
            )
            for row, p_holm in zip(
                (r for r in rows.values() if "p_permutation" in r), adjusted, strict=True
            ):
                row["p_holm"] = round(p_holm, 6)
            block[metric_name] = rows
        if task == "moscow":
            block["weighted_kappa"]["status"] = (
                "primary" if ORDINAL_MAPPING_VERIFIED else "exploratory: ordinal mapping unverified"
            )
            block["kappa_nominal"]["status"] = "primary"
        report["tasks"][task] = block
    return report


# ── reliability ────────────────────────────────────────────────────────────────────────


def reliability(runs: dict[str, AgencyRun]) -> dict[str, Any]:
    universe = set.union(*(set(run.items) for run in runs.values()))
    report: dict[str, Any] = {
        "universe_ids": len(universe),
        "per_protocol": {},
        "completion_mcnemar": {},
    }
    for name, run in runs.items():
        report["per_protocol"][name] = {
            "n_total": run.n_total,
            "completed": len(run.items),
            "completion_rate": round(len(run.items) / run.n_total, 4),
            "n_failed": run.results.get("n_failed"),
            "agents": {
                agent: {
                    "termination": block.get("termination"),
                    "failures_per_100_calls": block.get("failures_per_100_calls"),
                    "invalid_action_rate": block.get("invalid_action_rate"),
                }
                for agent, block in run.metrics.get("agent_activity", {}).items()
            },
        }
    for a, b in combinations(PROTOCOLS, 2):
        ordered = sorted(universe)
        test = mcnemar_exact(
            [i in runs[b].items for i in ordered], [i in runs[a].items for i in ordered]
        )
        report["completion_mcnemar"][f"{b}-{a}"] = {
            f"only_{b}_completed": test.only_a_correct,
            f"only_{a}_completed": test.only_b_correct,
            "p_value": round(test.p_value, 6),
        }
    return report


# ── cost ───────────────────────────────────────────────────────────────────────────────


def cost_report(runs: dict[str, AgencyRun]) -> dict[str, Any]:
    per_run = {name: cost_per_requirement(run.episodes) for name, run in runs.items()}
    by_phase = {name: cost_by_phase(run.episodes) for name, run in runs.items()}
    ids = sorted(set.intersection(*(set(costs) for costs in per_run.values())))
    if len(ids) < 2:
        return {"n_paired": len(ids), "by_phase": by_phase, "verdict": "insufficient_data"}
    report: dict[str, Any] = {"n_paired": len(ids), "by_phase": by_phase, "metrics": {}}
    for cost in COSTS:
        values = {name: [per_run[name][i][cost] for i in ids] for name in runs}
        rows = {
            f"{a}-{b}": {
                "mean_first": round(float(np.mean(values[a])), 3),
                "mean_second": round(float(np.mean(values[b])), 3),
                "mean_delta": round(float(np.mean(values[a])) - float(np.mean(values[b])), 3),
                "p_wilcoxon": round(wilcoxon_test(values[a], values[b]).p_value, 6),
            }
            for a, b in (("P2", "P0"), ("P3", "P0"), ("P3", "P2"))
        }
        adjusted = holm_adjust([row["p_wilcoxon"] for row in rows.values()])
        for row, p_holm in zip(rows.values(), adjusted, strict=True):
            row["p_holm"] = round(p_holm, 6)
        friedman = friedman_test([values[name] for name in PROTOCOLS])
        report["metrics"][cost] = {
            "friedman": {
                "statistic": round(friedman.statistic, 4),
                "p_value": round(friedman.p_value, 6),
            },
            "contrasts": rows,
        }
    return report


# ── effect of the interaction ──────────────────────────────────────────────────────────


def _agreement_changes(
    rows: list[dict[str, Any]], initial: str, final: str, ref: str
) -> dict[str, Any]:
    """Among the requirements whose decision changed, how many started disagreeing with the
    reference and ended agreeing (improved), the reverse (worsened), or neither (neutral).
    It counts agreement only, so it needs no ordering of the labels."""
    measured = [r for r in rows if r[ref] is not None and r[initial] is not None]
    changed = [r for r in measured if r[initial] != r[final]]
    improved = sum(1 for r in changed if r[final] == r[ref] and r[initial] != r[ref])
    worsened = sum(1 for r in changed if r[initial] == r[ref] and r[final] != r[ref])
    result: dict[str, Any] = {
        "measured": len(measured),
        "changed": len(changed),
        "improved": improved,
        "worsened": worsened,
        "neutral": len(changed) - improved - worsened,
    }
    if improved + worsened:
        test = mcnemar_exact(
            [True] * improved + [False] * worsened, [False] * improved + [True] * worsened
        )
        result["p_improved_vs_worsened"] = round(test.p_value, 6)
    return result


def interaction_effect(runs: dict[str, AgencyRun]) -> dict[str, Any]:
    """Did the exchange between the agents change decisions, and toward the benchmark's labels?
    An association within one run, not a causal estimate."""
    report: dict[str, Any] = {}
    for name, run in runs.items():
        if not run.decisions:
            report[name] = {"verdict": "no first-pass decisions recorded"}
            continue
        report[name] = {
            "requests": (run.metrics.get("interaction") or {}).get("requests", 0),
            "type": _agreement_changes(
                run.decisions, "initial_type", "final_type", "reference_type"
            ),
            "priority": _agreement_changes(
                run.decisions, "initial_priority", "final_priority", "reference_priority"
            ),
        }
    return report


# ── manipulation check ─────────────────────────────────────────────────────────────────


def manipulation_check(runs: dict[str, AgencyRun]) -> dict[str, Any]:
    """Did the protocols differ in what the agents could and did do? An action that was never on
    offer is an implementation problem; one that was on offer and unused is a result."""
    report: dict[str, Any] = {}
    for name, run in runs.items():
        offered: dict[str, set[str]] = {}
        for episode in run.episodes:
            key = f"{episode.agent}/{episode.phase.value}"
            offered.setdefault(key, set()).update(episode.available_actions)
        report[name] = {
            "offered_actions": {key: sorted(actions) for key, actions in sorted(offered.items())},
            "request_reclassification_offered": any(
                REQUEST_RECLASSIFICATION in actions for actions in offered.values()
            ),
            "interaction": run.metrics.get("interaction"),
            "message_flows": message_flows(run.messages),
            "mediator": run.metrics.get("mediator"),
            "action_sequence_entropy_bits": {
                agent: block["action_sequence_entropy_bits"]
                for agent, block in run.metrics.get("agent_activity", {}).items()
            },
        }
    return report


# ── quality versus cost ────────────────────────────────────────────────────────────────


def tradeoff(quality: dict[str, Any], cost: dict[str, Any]) -> list[dict[str, Any]]:
    """For every contrast and quality metric: the quality delta beside the cost deltas, and a
    plain reading. No weighted index: dominance is claimed only when no cost goes the wrong way."""
    if "metrics" not in cost:
        return []
    rows: list[dict[str, Any]] = []
    for task, metrics in quality["tasks"].items():
        for metric_name, contrasts in metrics.items():
            for contrast, row in contrasts.items():
                if not isinstance(row, dict) or "delta" not in row:
                    continue
                deltas = {c: cost["metrics"][c]["contrasts"][contrast]["mean_delta"] for c in COSTS}
                better = row["ci95"][0] > 0
                worse = row["ci95"][1] < 0
                if not (better or worse):
                    reading = "quality_indistinguishable"
                elif better and all(d <= 0 for d in deltas.values()):
                    reading = "first_dominates"
                elif worse and all(d >= 0 for d in deltas.values()):
                    reading = "second_dominates"
                else:
                    reading = "trade_off"
                rows.append(
                    {
                        "task": task,
                        "metric": metric_name,
                        "contrast": contrast,
                        "delta_quality": row["delta"],
                        "ci95": row["ci95"],
                        "delta_cost": deltas,
                        "reading": reading,
                    }
                )
    return rows


# ── noise floor ────────────────────────────────────────────────────────────────────────


def noise_floor(
    p0: AgencyRun,
    repeat: AgencyRun,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """P0 run twice under the same configuration. Any difference between the two is run-to-run
    noise, so it is the size a protocol effect has to exceed to mean anything."""
    pair = {"P0R": repeat, "P0": p0}
    quality = quality_contrasts(pair, n_permutations, n_bootstrap, seed, [NOISE_CONTRAST])
    cost = cost_per_requirement
    per_run = {name: cost(run.episodes) for name, run in pair.items()}
    ids = sorted(set(per_run["P0"]) & set(per_run["P0R"]))
    return {
        "quality": quality,
        "cost_mean_delta": {
            c: round(float(np.mean([per_run["P0R"][i][c] - per_run["P0"][i][c] for i in ids])), 3)
            for c in COSTS
        }
        if ids
        else None,
    }


def annotate_with_noise(quality: dict[str, Any], noise: dict[str, Any]) -> None:
    """Mark every protocol contrast with the noise-floor delta of the same metric and whether the
    protocol effect is larger in absolute value."""
    key = "-".join(NOISE_CONTRAST)
    for task, metrics in quality["tasks"].items():
        for metric_name, contrasts in metrics.items():
            floor = noise["quality"]["tasks"].get(task, {}).get(metric_name, {}).get(key, {})
            for row in contrasts.values():
                if not isinstance(row, dict) or "delta" not in row or "delta" not in floor:
                    continue
                row["noise_floor_delta"] = floor["delta"]
                row["exceeds_noise_floor"] = abs(row["delta"]) > abs(floor["delta"])


# ── the whole analysis ─────────────────────────────────────────────────────────────────


def analyze(
    runs: dict[str, AgencyRun],
    p0_repeat: AgencyRun | None = None,
    exploratory: bool = False,
    n_permutations: int | None = None,
    n_bootstrap: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    spec = load_spec()
    inference = spec["inference"]
    if p0_repeat is None and not exploratory:
        raise NotConfirmatoryError(
            "the confirmatory design needs the repeated P0 run (--p0-repeat)"
        )
    everything = {**runs, **({"P0R": p0_repeat} if p0_repeat else {})}
    ensure_confirmatory(everything, exploratory)
    ensure_same_design(everything)
    permutations = n_permutations or inference["n_permutations"]
    bootstrap = n_bootstrap or inference["n_bootstrap"]
    seed_value = spec["seed"] if seed is None else seed
    quality = quality_contrasts(runs, permutations, bootstrap, seed_value)
    noise = (
        noise_floor(runs["P0"], p0_repeat, permutations, bootstrap, seed_value)
        if p0_repeat
        else None
    )
    if noise:
        annotate_with_noise(quality, noise)
    cost = cost_report(runs)
    return {
        "architecture_claim": evaluate_arms(runs, spec["architecture_claim"]),
        "status": "exploratory" if exploratory else "confirmatory",
        "spec_sha256": spec_sha256(),
        "replicates": spec["replicates"],
        "runs": {name: run.path.name for name, run in everything.items()},
        "ordinal_mapping_verified": ORDINAL_MAPPING_VERIFIED,
        "quality": quality,
        "noise_floor": noise,
        "reliability": reliability(runs),
        "cost": cost,
        "interaction_effect": interaction_effect(runs),
        "manipulation_check": manipulation_check(runs),
        "tradeoff": tradeoff(quality, cost),
        "limitations": list(LIMITATIONS),
    }


# ── finding the arms ───────────────────────────────────────────────────────────────────

REPEAT_ARM = "P0R"


class ArmsNotFoundError(ValueError):
    """Raised when the results folder does not hold exactly one confirmatory run per arm."""


def _arm_of(manifest: dict[str, Any], spec: dict[str, Any]) -> str | None:
    strategy_to_arm = {strategy: arm for arm, strategy in spec["protocols"].items()}
    arm = strategy_to_arm.get(manifest.get("strategy", ""))
    tag = manifest.get("tag")
    if arm is None:
        return None
    if tag is None:
        return arm
    noise = spec["noise_floor"]
    return REPEAT_ARM if arm == noise["protocol"] and tag == noise["tag"] else None


def find_arms(results_dir: Path, spec: dict[str, Any] | None = None) -> dict[str, Path]:
    """The confirmatory run of each arm (P0, P2, P3 and the tagged P0 repeat) in a results folder.
    Exploratory runs, runs under another spec and the `pipeline` control are skipped."""
    spec = spec or load_spec()
    current = spec_sha256()
    found: dict[str, list[Path]] = {}
    for manifest_path in sorted(results_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = manifest.get("confirmatory") or {}
        if not status.get("is_confirmatory") or status.get("spec_sha256") != current:
            continue
        if arm := _arm_of(manifest, spec):
            found.setdefault(arm, []).append(manifest_path.parent)
    problems = [
        f"{arm}: " + (", ".join(p.name for p in found[arm]) if arm in found else "no run")
        for arm in (*PROTOCOLS, REPEAT_ARM)
        if len(found.get(arm, [])) != 1
    ]
    if problems:
        raise ArmsNotFoundError(
            "need exactly one confirmatory run per arm in "
            f"{results_dir}; found: " + "; ".join(problems)
        )
    return {arm: paths[0] for arm, paths in found.items()}


def check_arm_tags(runs: dict[str, AgencyRun], repeat: AgencyRun | None) -> None:
    """Explicit paths must not swap the repeat with the first P0 run."""
    noise_tag = load_spec()["noise_floor"]["tag"]
    for name, run in runs.items():
        if run.manifest.get("tag") is not None:
            raise NotConfirmatoryError(f"{name} ({run.path.name}) carries a tag; arms have none")
    if repeat is not None and repeat.manifest.get("tag") != noise_tag:
        raise NotConfirmatoryError(
            f"--p0-repeat ({repeat.path.name}) must have been run with --tag {noise_tag}"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in PROTOCOLS:
        parser.add_argument(f"--{name.lower()}", help=f"{name} run directory")
    parser.add_argument("--p0-repeat", help="second P0 run, the noise floor")
    parser.add_argument(
        "--results-dir",
        default="experiments/results",
        help="where to find the arms not given explicitly (confirmatory runs only)",
    )
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--permutations", type=int)
    parser.add_argument("--bootstrap", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    explicit = {name: getattr(args, name.lower()) for name in PROTOCOLS}
    repeat_path = args.p0_repeat
    if not all(explicit.values()) or (repeat_path is None and not args.exploratory):
        discovered = find_arms(Path(args.results_dir))
        explicit = {name: explicit[name] or discovered[name] for name in PROTOCOLS}
        repeat_path = repeat_path or discovered[REPEAT_ARM]
    runs = {name: load_run(Path(path)) for name, path in explicit.items()}
    repeat = load_run(Path(repeat_path)) if repeat_path else None
    if not args.exploratory:
        check_arm_tags(runs, repeat)
    result = analyze(runs, repeat, args.exploratory, args.permutations, args.bootstrap)
    manifest = build_analysis_manifest(
        "analyze_protocols",
        runs["P0"].information_regime,
        {name: str(run.path) for name, run in runs.items()},
        dataset_path=str(runs["P0"].manifest["dataset"]),
    )
    print(f"status: {result['status']} | paired: {result['quality']['n_paired']}")
    print(f"written: {write_analysis_result('protocols', manifest, result)}")


if __name__ == "__main__":
    main()
