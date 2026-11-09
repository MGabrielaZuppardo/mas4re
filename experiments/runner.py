from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from datasets.registry import PROMISE, build_adapter, default_data_lang
from domain.agent_actions import AgentEpisode
from domain.enums import CoordinationProtocol, InformationRegime, Lang
from domain.models import ClassifiedRequirement, PipelineState
from evaluation.agent_activity import (
    ACTIONS_FILENAME,
    MESSAGES_FILENAME,
    summarize_episodes,
    summarize_messages,
    write_episodes_jsonl,
    write_messages_jsonl,
)
from evaluation.backlog import (
    BACKLOG_FILENAME,
    DEFAULT_DELIMITER,
    FAILED_FILENAME,
    build_backlog,
    write_backlog_csv,
    write_failed_csv,
)
from evaluation.decisions import DECISIONS_FILENAME, build_decisions, write_decisions_jsonl
from evaluation.interaction_failures import summarize_interaction_failures
from evaluation.metrics.classification import (
    compute_classification_metrics,
    compute_subcategory_metrics,
)
from evaluation.metrics.prioritization import (
    compute_moscow_distribution,
    compute_priority_reference_metrics,
)
from evaluation.protocol_conformance import summarize_conformance
from evaluation.trace_writer import TraceWriter
from experiments.confirmatory import assess
from experiments.strategy import OrchestrationStrategy
from pipeline.protocol_spec import PROTOCOL_SPECS

logger = logging.getLogger(__name__)

_TRACE_DIR = Path("experiments") / "traces"
_RUN_DATE_FORMAT = "%Y%m%d-%H%M%S"
_TAG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def check_tag(tag: str | None) -> str | None:
    """A tag becomes part of the run folder name, so it is kept to lowercase words and hyphens."""
    if tag is not None and not _TAG.match(tag):
        raise ValueError(f"tag must be lowercase letters, digits and hyphens, got {tag!r}")
    return tag


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _file_md5(path: Path) -> str:
    if not path.exists():
        return "unknown"
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def _langgraph_version() -> str:
    try:
        from importlib.metadata import version

        return version("langgraph")
    except Exception:
        return "unknown"


@dataclass
class RunConfig:
    """Frozen parameters of one experimental run (SQ2 reproducibility).

    `lang` is the language of the prompts; `data_lang` is the language of the requirement text
    the agents read. They are independent: a run may give Spanish text to Portuguese prompts.
    """

    strategy_name: str
    model: str
    dataset_path: str
    lang: str = "pt"
    n_samples: int | None = None
    seed: int = 42
    temperature: float = 0.0
    prompt_version: str = "v1"
    information_regime: InformationRegime = InformationRegime.ZERO_SHOT
    dataset_name: str = PROMISE
    data_lang: str | None = None
    tag: str | None = None
    max_steps: int | None = None
    exploratory: dict[str, Any] | None = None

    @property
    def models(self) -> dict[str, str]:
        """Model of each agent (`model` is `classifier+prioritizer`, or a single name)."""
        classifier, _, prioritizer = self.model.partition("+")
        return {"classifier": classifier, "prioritizer": prioritizer or classifier}

    def __post_init__(self) -> None:
        check_tag(self.tag)
        self.information_regime = InformationRegime(self.information_regime)
        self.data_lang = Lang(self.data_lang or default_data_lang(self.dataset_name)).value


@dataclass
class RunResult:
    config: RunConfig
    state: PipelineState
    elapsed_seconds: float
    run_id: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    backlog_path: Path | None = None
    failed_path: Path | None = None
    actions_path: Path | None = None
    messages_path: Path | None = None


class ExperimentRunner:
    """Runs a strategy under a frozen config and writes a reproducible
    manifest. Same runner for every architecture so SQ2 comparisons are
    controlled.

    Each run produces these artifacts in experiments/results/{run_id}/:
        manifest.json  — frozen parameters + git commit + dataset hash
        results.json   — quality metrics + full predictions
        backlog.csv    — classification + priority of every processed requirement
        failed.csv     — requirements the agents could not process (only if there are any)
        actions.jsonl  — one episode per agent and requirement: goal, beliefs, the actions the
                         agent chose, tokens (only for strategies with agents that act)
        decisions.jsonl — the first-pass and the final type and priority of every requirement,
                         with the reference labels (only for agents that can revise a decision)
        messages.jsonl — every typed message the agents (or the mediator) exchanged, with its
                         evidence and what it replies to (only when there was any)
        trace.jsonl    — per-requirement latency + parse outcome (TraceWriter, in trace_dir)
    """

    def __init__(
        self,
        out_dir: str = "experiments/results",
        trace_dir: str | None = None,
        backlog_delimiter: str = DEFAULT_DELIMITER,
        include_reference: bool = True,
    ) -> None:
        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._trace_dir = Path(trace_dir) if trace_dir is not None else _TRACE_DIR
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._backlog_delimiter = backlog_delimiter
        self._include_reference = include_reference

    def _build_manifest(
        self,
        config: RunConfig,
        run_id: str,
        n_loaded: int,
        elapsed: float,
        trace_path: Path,
        started_at: datetime,
        finished_at: datetime,
        agent_runtime: dict[str, Any] | None = None,
        protocol: str | None = None,
        architecture: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = {
            "run_id": run_id,
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "strategy": config.strategy_name,
            "model": config.model,
            "models": config.models,
            "lang": config.lang,
            "seed": config.seed,
            "temperature": config.temperature,
            "prompt_version": config.prompt_version,
            "information_regime": config.information_regime.value,
            "dataset_name": config.dataset_name,
            "data_lang": config.data_lang,
            "tag": config.tag,
            "max_steps": config.max_steps,
            "dataset": config.dataset_path,
            "dataset_n": n_loaded,
            "dataset_md5": _file_md5(Path(config.dataset_path)),
            "langgraph_version": _langgraph_version(),
            "elapsed_seconds": round(elapsed, 3),
            "trace_path": str(trace_path),
            "confirmatory": assess(
                config.strategy_name,
                {
                    "dataset_name": config.dataset_name,
                    "data_lang": config.data_lang,
                    "lang": config.lang,
                    "seed": config.seed,
                    "temperature": config.temperature,
                    "model": config.model,
                    "information_regime": config.information_regime.value,
                },
                config.n_samples,
                n_loaded,
                tag=config.tag,
                max_steps=config.max_steps,
                exploratory_variant=(config.exploratory or {}).get("variant"),
            ),
        }
        if agent_runtime:
            manifest["agent_runtime"] = agent_runtime
        if protocol:
            manifest["protocol"] = protocol
        if architecture:
            manifest["architecture"] = architecture
        if config.exploratory:
            manifest["exploratory"] = config.exploratory
        return manifest

    def _compute_metrics(self, state: PipelineState) -> dict[str, Any]:
        ground_truth = state.raw_requirements
        classified = cast(
            "list[ClassifiedRequirement]",
            state.prioritized_requirements or state.classified_requirements,
        )

        metrics: dict[str, Any] = {}
        if classified:
            metrics["classification"] = compute_classification_metrics(classified, ground_truth)
            metrics["subcategory"] = compute_subcategory_metrics(classified, ground_truth)
        if state.prioritized_requirements:
            metrics["moscow_distribution"] = compute_moscow_distribution(
                state.prioritized_requirements
            )
            if gold := compute_priority_reference_metrics(
                state.prioritized_requirements, ground_truth
            ):
                metrics["prioritization"] = gold
        if "inter_agent_conflicts" in state.metrics:
            metrics["inter_agent_conflicts"] = state.metrics["inter_agent_conflicts"]
        if "failure_detections" in state.metrics:
            metrics["failure_detections"] = state.metrics["failure_detections"]
        if "mediation" in state.metrics:
            metrics["mediation"] = state.metrics["mediation"]
        if episodes := self._episodes_of(state):
            metrics["agent_activity"] = summarize_episodes(episodes)
        if state.messages:
            metrics["interaction"] = summarize_messages(state.messages)
        if protocol := state.metrics.get("protocol"):
            spec = PROTOCOL_SPECS[CoordinationProtocol(protocol)]
            metrics["conformance"] = summarize_conformance(spec, state.messages)
            metrics["interaction_failures"] = summarize_interaction_failures(
                state.messages,
                build_decisions(
                    state.metrics.get("initial_decisions", {}), state.prioritized_requirements
                ),
            )
        if "mediator" in state.metrics:
            metrics["mediator"] = state.metrics["mediator"]
        return metrics

    @staticmethod
    def _episodes_of(state: PipelineState) -> list[AgentEpisode]:
        return [AgentEpisode.model_validate(e) for e in state.metrics.get("agent_episodes", [])]

    def execute(self, strategy: OrchestrationStrategy, config: RunConfig) -> RunResult:
        adapter = build_adapter(config.dataset_name, config.dataset_path, Lang(config.data_lang))
        requirements = (
            adapter.load_sample(config.n_samples, seed=config.seed)
            if config.n_samples
            else adapter.load()
        )

        # Build run_id early so TraceWriter and manifest share the same key.
        started_at = datetime.now().astimezone()
        model_slug = re.sub(r"[^A-Za-z0-9._-]", "-", config.model)
        tag_part = f"_{config.tag}" if config.tag else ""
        run_id = (
            f"{strategy.name}_{model_slug}_{config.data_lang}-{config.lang}"
            f"_n{len(requirements)}{tag_part}_{started_at.strftime(_RUN_DATE_FORMAT)}"
        )
        run_path = self._out / run_id
        run_path.mkdir(parents=True, exist_ok=False)
        trace_path = self._trace_dir / f"{run_id}.jsonl"

        logger.info(
            "Run start | run_id=%s | strategy=%s | model=%s | lang=%s | n=%d",
            run_id,
            strategy.name,
            config.model,
            config.lang,
            len(requirements),
        )

        start = time.perf_counter()
        with TraceWriter(output_dir=self._trace_dir, run_id=run_id) as tw:
            state = strategy.execute(requirements, trace_writer=tw)
        elapsed = time.perf_counter() - start
        finished_at = datetime.now().astimezone()

        manifest = self._build_manifest(
            config,
            run_id,
            len(requirements),
            elapsed,
            trace_path,
            started_at,
            finished_at,
            state.metrics.get("agent_runtime"),
            state.metrics.get("protocol"),
            state.metrics.get("architecture"),
        )
        (run_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        metrics = self._compute_metrics(state)
        predictions = [r.model_dump(mode="json") for r in state.prioritized_requirements] or [
            r.model_dump(mode="json") for r in state.classified_requirements
        ]
        backlog = build_backlog(state, run_id)
        backlog_path = write_backlog_csv(
            backlog.rows,
            run_path / BACKLOG_FILENAME,
            self._backlog_delimiter,
            self._include_reference,
        )
        failed_path = write_failed_csv(
            backlog.failed,
            run_path / FAILED_FILENAME,
            self._backlog_delimiter,
            self._include_reference,
        )
        actions_path = write_episodes_jsonl(self._episodes_of(state), run_path / ACTIONS_FILENAME)
        messages_path = write_messages_jsonl(state.messages, run_path / MESSAGES_FILENAME)
        write_decisions_jsonl(
            build_decisions(
                state.metrics.get("initial_decisions", {}), state.prioritized_requirements
            )
            if "initial_decisions" in state.metrics
            else [],
            run_path / DECISIONS_FILENAME,
        )
        results = {
            "run_id": run_id,
            "executed_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "config": {
                "strategy": config.strategy_name,
                "model": config.model,
                "lang": config.lang,
                "seed": config.seed,
                "n": len(requirements),
                "information_regime": config.information_regime.value,
                "dataset_name": config.dataset_name,
                "data_lang": config.data_lang,
            },
            "metrics": metrics,
            "n_predictions": len(predictions),
            "n_failed": len(backlog.failed),
            "backlog_path": str(backlog_path),
            "failed_path": str(failed_path) if failed_path else None,
            "actions_path": str(actions_path) if actions_path else None,
            "messages_path": str(messages_path) if messages_path else None,
            "predictions": predictions,
        }
        (run_path / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info(
            "Run done | run_id=%s | elapsed=%.2fs | classification=%s",
            run_id,
            elapsed,
            metrics.get("classification", {}),
        )
        return RunResult(
            config=config,
            state=state,
            elapsed_seconds=elapsed,
            run_id=run_id,
            manifest=manifest,
            metrics=metrics,
            backlog_path=backlog_path,
            failed_path=failed_path,
            actions_path=actions_path,
            messages_path=messages_path,
        )
