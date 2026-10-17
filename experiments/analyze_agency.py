"""Agency on real runs: criteria C1, C2 and C4 consolidated from the logs, C5 and C7 tested.

The criteria are defined in docs/agency_conformance.md. C1, C2 and C4 are read straight from what
the agents did (action sequences, budget, answers to requests). The other two need statistics:

* C5 -- does an observation change the next action? For each observing tool, a contingency table
  of the class of what was observed against the class of the action that followed, with Fisher's
  exact test (2x2) or chi-square and Cramer's V when the counts allow, and "insufficient_data"
  when they do not.
* C7 -- ablation against the workflow: `pipeline` against `pipeline_agentic*` on the same items,
  paired by requirement id, with cost and coverage always beside the quality figures.

Everything is computed inside one language cell (dataset, data language, prompt language); runs
of different cells are never combined, and a workflow run is only paired with an agentic run that
shares model, seed, temperature and dataset. ES/ES on PrioReSpa is the primary cell; the others are
reported as sensitivity.

Usage (from the repository root):
    python -m experiments.analyze_agency --agentic RUN_DIR [--agentic RUN_DIR ...]
        [--workflow RUN_DIR] [--bootstrap 2000] [--seed 42]

Output:
    experiments/results/analysis/agency_<timestamp>.json
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import cohen_kappa_score

from agents.agentic.toolkit import (
    BACKLOG_OVERVIEW,
    CHECK_CONSISTENCY,
    CONSISTENT,
    NO_SIMILAR_CASES,
    RECALL_SIMILAR_CASES,
)
from domain.agent_actions import ActionKind, ActionRecord, AgentEpisode, EpisodePhase
from domain.enums import InformationRegime, MoSCoWPriority
from domain.models import PrioritizedRequirement
from evaluation.agent_activity import ACTIONS_FILENAME, summarize_episodes
from evaluation.metrics.statistical import (
    bootstrap_ci,
    bootstrap_statistic_difference,
    mcnemar_exact,
)
from experiments.analysis_common import (
    MixedInformationRegimeError,
    build_analysis_manifest,
    read_json,
    write_analysis_result,
)

ALPHA = 0.05
MIN_OBSERVATIONS = 20
MIN_EXPECTED_COUNT = 5
DEFAULT_BOOTSTRAP = 2_000
DEFAULT_SEED = 42
MUST_SHARE_HIGH = 0.5
WORKFLOW_STRATEGY = "pipeline"

_RANKS = sorted(priority.ordinal for priority in MoSCoWPriority)
_COMPARABLE = ("model", "seed", "temperature", "dataset_md5", "dataset_n")
_DECIDED_SO_FAR = re.compile(r"decided so far[^:]*: (?P<decided>.+?)\.\s*$")

CALLS_ANOTHER_TOOL = "calls_another_tool"


class MixedCellError(ValueError):
    """Raised when one analysis would combine runs from different language cells."""


class UnpairableRunsError(ValueError):
    """Raised when two runs differ in something other than the coordination strategy."""


# ── runs ───────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    """Where a run sits in the language design: the dataset, the language of its text and the
    language of the prompts."""

    dataset: str
    data_lang: str
    prompt_lang: str

    @property
    def status(self) -> str:
        return "primary" if self == PRIMARY_CELL else "sensitivity"

    def as_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "data_lang": self.data_lang,
            "prompt_lang": self.prompt_lang,
            "status": self.status,
        }


PRIMARY_CELL = Cell(dataset="priorespa", data_lang="es", prompt_lang="es")


@dataclass(frozen=True)
class AgencyRun:
    path: Path
    manifest: dict[str, Any]
    results: dict[str, Any]
    items: dict[str, PrioritizedRequirement]
    episodes: list[AgentEpisode]

    @property
    def strategy(self) -> str:
        return str(self.manifest["strategy"])

    @property
    def protocol(self) -> str | None:
        return self.manifest.get("protocol")

    @property
    def cell(self) -> Cell:
        return Cell(
            self.manifest["dataset_name"], self.manifest["data_lang"], self.manifest["lang"]
        )

    @property
    def information_regime(self) -> InformationRegime:
        return InformationRegime(self.manifest.get("information_regime", "zero_shot"))

    @property
    def n_total(self) -> int:
        return int(self.manifest["dataset_n"])

    @property
    def metrics(self) -> dict[str, Any]:
        return self.results.get("metrics", {})


def load_run(run_dir: str | Path) -> AgencyRun:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    missing = [k for k in ("dataset_name", "data_lang", "lang", "dataset_n") if k not in manifest]
    if missing:
        raise ValueError(f"{run_dir}: manifest without {missing}; the run predates language cells")
    results = read_json(run_dir / "results.json")
    items = {p["id"]: PrioritizedRequirement.model_validate(p) for p in results["predictions"]}
    actions = run_dir / ACTIONS_FILENAME
    lines = actions.read_text(encoding="utf-8").splitlines() if actions.exists() else []
    episodes = [AgentEpisode.model_validate_json(line) for line in lines if line.strip()]
    return AgencyRun(run_dir, manifest, results, items, episodes)


def single_cell(runs: Sequence[AgencyRun]) -> Cell:
    cells = {run.cell for run in runs}
    if len(cells) != 1:
        found = sorted(f"{c.dataset}/{c.data_lang}/{c.prompt_lang}" for c in cells)
        raise MixedCellError(f"Refusing to analyse runs from different language cells: {found}")
    return cells.pop()


def single_regime(runs: Sequence[AgencyRun]) -> InformationRegime:
    regimes = {run.information_regime for run in runs}
    if len(regimes) != 1:
        found = sorted(r.value for r in regimes)
        raise MixedInformationRegimeError(
            f"Refusing to analyse runs from different information regimes together: {found}"
        )
    return regimes.pop()


def ensure_pairable(workflow: AgencyRun, agentic: AgencyRun) -> None:
    differing = [
        f"{field}: {workflow.manifest.get(field)!r} != {agentic.manifest.get(field)!r}"
        for field in _COMPARABLE
        if workflow.manifest.get(field) != agentic.manifest.get(field)
    ]
    if differing:
        raise UnpairableRunsError(
            f"{workflow.path.name} and {agentic.path.name} differ in more than the strategy: "
            + "; ".join(differing)
        )


# ── C1, C2, C4: read from what the agents did ──────────────────────────────────────────


def agent_criteria(run: AgencyRun) -> dict[str, Any]:
    per_agent = {
        agent: {
            "episodes": block["episodes"],
            "distinct_action_sequences": block["distinct_action_sequences"],
            "action_sequence_entropy_bits": block["action_sequence_entropy_bits"],
            "entropy_above_zero": block["action_sequence_entropy_bits"] > 0,
            "budget_exhaustion_rate": block["budget_exhaustion_rate"],
            "invalid_action_rate": block["invalid_action_rate"],
        }
        for agent, block in summarize_episodes(run.episodes).items()
    }
    interaction = run.metrics.get("interaction", {})
    rates = (interaction.get("agree_rate"), interaction.get("refuse_rate"))
    return {
        "c1_c2": per_agent,
        "c4": {
            **interaction,
            "both_rates_positive": None if None in rates else all(rate > 0 for rate in rates),
        },
    }


# ── C5: does an observation change the next action? ────────────────────────────────────


def _consistency_class(action: ActionRecord) -> str:
    return "consistent" if action.observation.startswith(CONSISTENT) else "inconsistent"


def _recall_class(action: ActionRecord) -> str:
    return "no_similar_cases" if action.observation == NO_SIMILAR_CASES else "similar_cases_found"


def _overview_class(action: ActionRecord) -> str:
    match = _DECIDED_SO_FAR.search(action.observation)
    if match is None or match["decided"] == "none yet":
        return "no_priorities_yet"
    counts = {
        key.strip(): int(value)
        for key, value in (part.split(":") for part in match["decided"].split(","))
    }
    share = counts.get(MoSCoWPriority.MUST_HAVE.value, 0) / sum(counts.values())
    return "must_share_high" if share >= MUST_SHARE_HIGH else "must_share_low"


_OBSERVERS: dict[str, Callable[[ActionRecord], str]] = {
    CHECK_CONSISTENCY: _consistency_class,
    RECALL_SIMILAR_CASES: _recall_class,
    BACKLOG_OVERVIEW: _overview_class,
}


def _answer_of(action: ActionRecord) -> tuple[str, str]:
    arguments = action.arguments
    return (
        str(arguments.get("requirement_type", "")).upper(),
        str(arguments.get("nfr_category") or "").upper(),
    )


def _follow_up_class(observer: ActionRecord, following: ActionRecord) -> str:
    if following.kind is not ActionKind.SUBMIT:
        return CALLS_ANOTHER_TOOL
    if observer.name == CHECK_CONSISTENCY:
        same = _answer_of(observer) == _answer_of(following)
        return "submits_checked_answer" if same else "submits_different_answer"
    if observer.name == BACKLOG_OVERVIEW:
        must = following.arguments.get("priority") == MoSCoWPriority.MUST_HAVE.value
        return "submits_must" if must else "submits_not_must"
    return "submits"


@dataclass(frozen=True)
class Observation:
    agent: str
    tool: str
    observed: str
    followed: str


def observations(episodes: Iterable[AgentEpisode]) -> list[Observation]:
    """Every observation an agent made while working on an item, with what it did next. Only
    first-time work counts: replies and reconsiderations offer different tools."""
    found: list[Observation] = []
    for episode in episodes:
        if episode.phase is not EpisodePhase.WORK:
            continue
        actions = [a for a in episode.actions if a.kind is not ActionKind.INVALID]
        for current, following in zip(actions, actions[1:], strict=False):
            classify = _OBSERVERS.get(current.name) if current.kind is ActionKind.TOOL else None
            if classify is not None:
                found.append(
                    Observation(
                        episode.agent,
                        current.name,
                        classify(current),
                        _follow_up_class(current, following),
                    )
                )
    return found


def association(table: dict[str, Counter[str]]) -> dict[str, Any]:
    """Is the follow-up action associated with what was observed? Fisher's exact test for a 2x2
    table, chi-square otherwise (Cramer's V as effect size). Too few observations, a single class
    on either side, or small expected counts in a larger table give no verdict."""
    rows = [name for name, counts in table.items() if sum(counts.values())]
    columns = sorted({column for counts in table.values() for column in counts})
    n = sum(sum(counts.values()) for counts in table.values())
    report: dict[str, Any] = {
        "table": {row: dict(sorted(table[row].items())) for row in sorted(rows)},
        "n": n,
        "test": None,
        "p_value": None,
        "cramers_v": None,
        "verdict": "insufficient_data",
        "reason": None,
    }
    if n < MIN_OBSERVATIONS:
        report["reason"] = f"fewer than {MIN_OBSERVATIONS} observations"
        return report
    if len(rows) < 2 or len(columns) < 2:
        report["reason"] = "needs at least two observed classes and two follow-up classes"
        return report

    matrix = np.array([[table[row][column] for column in columns] for row in rows])
    chi2, p_chi2, _, expected = chi2_contingency(matrix, correction=False)
    if matrix.shape == (2, 2):
        report.update(test="fisher_exact", p_value=float(fisher_exact(matrix)[1]))
    elif (expected < MIN_EXPECTED_COUNT).any():
        report["reason"] = f"expected counts below {MIN_EXPECTED_COUNT} in a table larger than 2x2"
        return report
    else:
        report.update(test="chi2", p_value=float(p_chi2))
    report["cramers_v"] = round(math.sqrt(chi2 / (n * (min(matrix.shape) - 1))), 4)
    report["p_value"] = round(report["p_value"], 6)
    report["verdict"] = "associated" if report["p_value"] < ALPHA else "not_associated"
    return report


def observation_action_association(episodes: Iterable[AgentEpisode]) -> dict[str, Any]:
    """{agent: {tool: association}} for the tools that observe."""
    tables: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for seen in observations(episodes):
        tables[(seen.agent, seen.tool)][seen.observed][seen.followed] += 1
    report: dict[str, dict[str, Any]] = defaultdict(dict)
    for (agent, tool), table in sorted(tables.items()):
        report[agent][tool] = association(table)
    return dict(report)


# ── C7: workflow against agentic, on the same items ────────────────────────────────────


def _type_correct(item: PrioritizedRequirement) -> bool:
    return item.requirement_type.value == item.metadata.get("label_type")


def _delta(agentic: float, workflow: float) -> float:
    return round(agentic - workflow, 6)


def _accuracy_block(
    workflow_hits: Sequence[bool], agentic_hits: Sequence[bool], n_bootstrap: int, seed: int
) -> dict[str, Any]:
    """Accuracy of each side on the paired items and the agentic-minus-workflow difference, with
    a bootstrap CI and the exact McNemar test."""
    block: dict[str, Any] = {
        "workflow": round(float(np.mean(workflow_hits)), 4) if workflow_hits else None,
        "agentic": round(float(np.mean(agentic_hits)), 4) if agentic_hits else None,
    }
    if len(workflow_hits) < 2:
        return {**block, "verdict": "insufficient_data"}
    interval = bootstrap_ci(
        [float(h) for h in agentic_hits],
        [float(h) for h in workflow_hits],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    mcnemar = mcnemar_exact(agentic_hits, workflow_hits)
    return {
        **block,
        "delta": interval.mean_difference,
        "ci95": [interval.ci_lower, interval.ci_upper],
        "mcnemar": {
            "only_agentic_correct": mcnemar.only_a_correct,
            "only_workflow_correct": mcnemar.only_b_correct,
            "p_value": round(mcnemar.p_value, 6),
        },
    }


def _gold_ordinals(run: AgencyRun, ids: Sequence[str]) -> np.ndarray:
    return np.array([_gold(run.items[i]) for i in ids])


def _gold(item: PrioritizedRequirement) -> int:
    gold = item.gold_priority
    assert gold is not None
    return gold.ordinal


def _predicted_ordinals(run: AgencyRun, ids: Sequence[str]) -> np.ndarray:
    return np.array([_predicted(run.items[i]) for i in ids])


def _predicted(item: PrioritizedRequirement) -> int:
    assert item.priority is not None
    return item.priority.ordinal


def _weighted_kappa(gold: np.ndarray, predicted: np.ndarray) -> float:
    return float(cohen_kappa_score(gold, predicted, weights="linear", labels=_RANKS))


def _moscow_comparison(
    workflow: AgencyRun, agentic: AgencyRun, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    ids = sorted(
        i
        for i in workflow.items.keys() & agentic.items.keys()
        if workflow.items[i].gold_priority is not None
        and workflow.items[i].priority is not None
        and agentic.items[i].priority is not None
    )
    result: dict[str, Any] = {"n_paired": len(ids)}
    if len(ids) < 2:
        return {**result, "verdict": "insufficient_data"}

    gold = _gold_ordinals(workflow, ids)
    by_workflow, by_agentic = _predicted_ordinals(workflow, ids), _predicted_ordinals(agentic, ids)
    result["accuracy"] = _accuracy_block(
        list(by_workflow == gold), list(by_agentic == gold), n_bootstrap, seed
    )
    kappa_gap = bootstrap_statistic_difference(
        lambda idx: (
            _weighted_kappa(gold[idx], by_agentic[idx])
            - _weighted_kappa(gold[idx], by_workflow[idx])
        ),
        n=len(ids),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    result["weighted_kappa"] = {
        "workflow": round(_weighted_kappa(gold, by_workflow), 4),
        "agentic": round(_weighted_kappa(gold, by_agentic), 4),
        "delta": kappa_gap.mean_difference,
        "ci95": [kappa_gap.ci_lower, kappa_gap.ci_upper],
        "valid_resamples": kappa_gap.n_bootstrap,
    }
    return result


def _strict_accuracy(run: AgencyRun, hit: Callable[[PrioritizedRequirement], bool]) -> float:
    """Accuracy over every item of the dataset sample: an item the run never processed counts as
    wrong, so a run cannot look better by failing on its hard cases."""
    return round(sum(hit(item) for item in run.items.values()) / run.n_total, 4)


def _cost(run: AgencyRun) -> dict[str, Any]:
    activity = run.metrics.get("agent_activity")
    elapsed = float(run.manifest["elapsed_seconds"])
    return {
        "n_total": run.n_total,
        "n_processed": len(run.items),
        "n_failed": run.results.get("n_failed"),
        "coverage": round(len(run.items) / run.n_total, 4),
        "seconds_per_item": round(elapsed / run.n_total, 3),
        "llm_calls_per_item": (
            round(sum(a["llm_calls"] for a in activity.values()) / run.n_total, 3)
            if activity
            else None
        ),
        "tokens_per_item": (
            round(
                sum(a["input_tokens"] + a["output_tokens"] for a in activity.values())
                / run.n_total,
                1,
            )
            if activity
            else None
        ),
    }


def compare_to_workflow(
    workflow: AgencyRun,
    agentic: AgencyRun,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    ensure_pairable(workflow, agentic)
    paired = sorted(workflow.items.keys() & agentic.items.keys())
    type_comparison = _accuracy_block(
        [_type_correct(workflow.items[i]) for i in paired],
        [_type_correct(agentic.items[i]) for i in paired],
        n_bootstrap,
        seed,
    )
    type_comparison["strict"] = {
        "workflow": _strict_accuracy(workflow, _type_correct),
        "agentic": _strict_accuracy(agentic, _type_correct),
    }
    workflow_cost, agentic_cost = _cost(workflow), _cost(agentic)
    return {
        "workflow": workflow.path.name,
        "agentic": agentic.path.name,
        "strategy": agentic.strategy,
        "protocol": agentic.protocol,
        "n_paired": len(paired),
        "only_workflow": len(workflow.items.keys() - agentic.items.keys()),
        "only_agentic": len(agentic.items.keys() - workflow.items.keys()),
        "type_accuracy": type_comparison,
        "moscow": _moscow_comparison(workflow, agentic, n_bootstrap, seed),
        "cost": {
            "workflow": workflow_cost,
            "agentic": agentic_cost,
            "seconds_ratio": round(
                agentic_cost["seconds_per_item"] / workflow_cost["seconds_per_item"], 2
            )
            if workflow_cost["seconds_per_item"]
            else None,
        },
    }


# ── the whole analysis ─────────────────────────────────────────────────────────────────


def analyze(
    agentic_runs: Sequence[AgencyRun],
    workflow: AgencyRun | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    runs = [*agentic_runs, *([workflow] if workflow else [])]
    cell = single_cell(runs)
    single_regime(runs)
    return {
        "cell": cell.as_dict(),
        "runs": [
            {
                "run": run.path.name,
                "strategy": run.strategy,
                "protocol": run.protocol,
                "model": run.manifest["model"],
                **agent_criteria(run),
                "c5": observation_action_association(run.episodes),
            }
            for run in agentic_runs
        ],
        "c7": [compare_to_workflow(workflow, run, n_bootstrap, seed) for run in agentic_runs]
        if workflow
        else [],
    }


def _print_summary(result: dict[str, Any]) -> None:
    cell = result["cell"]
    print(f"cell: {cell['dataset']}/{cell['data_lang']}/{cell['prompt_lang']} ({cell['status']})")
    for run in result["runs"]:
        print(f"\n{run['run']}  [{run['strategy']}]")
        for agent, block in run["c1_c2"].items():
            print(
                f"  {agent:<11} entropy={block['action_sequence_entropy_bits']:.3f} bits  "
                f"budget_exhaustion={block['budget_exhaustion_rate']:.1%}  "
                f"invalid={block['invalid_action_rate']:.1%}"
            )
        c4 = run["c4"]
        print(
            f"  C4 requests={c4.get('requests', 0)}  agree_rate={c4.get('agree_rate')}  "
            f"refuse_rate={c4.get('refuse_rate')}"
        )
        for agent, tools in run["c5"].items():
            for tool, report in tools.items():
                print(f"  C5 {agent}/{tool}: {report['verdict']} (n={report['n']})")
    for comparison in result["c7"]:
        moscow = comparison["moscow"].get("accuracy", {})
        print(
            f"\nC7 {comparison['strategy']} vs workflow: paired={comparison['n_paired']}  "
            f"type_delta={comparison['type_accuracy'].get('delta')}  "
            f"moscow_delta={moscow.get('delta')}"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agentic", action="append", required=True, metavar="RUN_DIR")
    parser.add_argument("--workflow", metavar="RUN_DIR", help="a `pipeline` run on the same items")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    agentic_runs = [load_run(path) for path in args.agentic]
    workflow = load_run(args.workflow) if args.workflow else None
    if workflow is not None and workflow.strategy != WORKFLOW_STRATEGY:
        raise SystemExit(f"--workflow must be a '{WORKFLOW_STRATEGY}' run, got {workflow.strategy}")

    result = analyze(agentic_runs, workflow, args.bootstrap, args.seed)
    _print_summary(result)

    reference = agentic_runs[0]
    manifest = build_analysis_manifest(
        "analyze_agency",
        reference.information_regime,
        {
            "agentic": [str(run.path) for run in agentic_runs],
            "workflow": str(workflow.path) if workflow else None,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "alpha": ALPHA,
            "min_observations": MIN_OBSERVATIONS,
        },
        dataset_path=str(reference.manifest["dataset"]),
    )
    print(f"\nwritten: {write_analysis_result('agency', manifest, result)}")


if __name__ == "__main__":
    main()
