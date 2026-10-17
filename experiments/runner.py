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
from domain.enums import InformationRegime, Lang
from domain.models import ClassifiedRequirement, PipelineState
from evaluation.agent_activity import (
    ACTIONS_FILENAME,
    summarize_episodes,
    summarize_messages,
    write_episodes_jsonl,
)
from evaluation.backlog import (
    BACKLOG_FILENAME,
    DEFAULT_DELIMITER,
    FAILED_FILENAME,
    build_backlog,
    write_backlog_csv,
    write_failed_csv,
)
from evaluation.metrics.classification import (
    compute_classification_metrics,
    compute_subcategory_metrics,
)
from evaluation.metrics.prioritization import (
    compute_moscow_distribution,
    compute_priority_gold_metrics,
)
from evaluation.trace_writer import TraceWriter
from experiments.strategy import OrchestrationStrategy

logger = logging.getLogger(__name__)

_TRACE_DIR = Path("experiments") / "traces"


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

    def __post_init__(self) -> None:
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
        trace.jsonl    — per-requirement latency + parse outcome (TraceWriter, in trace_dir)
    """

    def __init__(
        self,
        out_dir: str = "experiments/results",
        trace_dir: str | None = None,
        backlog_delimiter: str = DEFAULT_DELIMITER,
        include_ground_truth: bool = True,
    ) -> None:
        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._trace_dir = Path(trace_dir) if trace_dir is not None else _TRACE_DIR
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._backlog_delimiter = backlog_delimiter
        self._include_ground_truth = include_ground_truth

    def _build_manifest(
        self,
        config: RunConfig,
        run_id: str,
        n_loaded: int,
        elapsed: float,
        trace_path: Path,
        agent_runtime: dict[str, Any] | None = None,
        protocol: str | None = None,
    ) -> dict[str, Any]:
        manifest = {
            "run_id": run_id,
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "strategy": config.strategy_name,
            "model": config.model,
            "lang": config.lang,
            "seed": config.seed,
            "temperature": config.temperature,
            "prompt_version": config.prompt_version,
            "information_regime": config.information_regime.value,
            "dataset_name": config.dataset_name,
            "data_lang": config.data_lang,
            "dataset": config.dataset_path,
            "dataset_n": n_loaded,
            "dataset_md5": _file_md5(Path(config.dataset_path)),
            "langgraph_version": _langgraph_version(),
            "elapsed_seconds": round(elapsed, 3),
            "trace_path": str(trace_path),
        }
        if agent_runtime:
            manifest["agent_runtime"] = agent_runtime
        if protocol:
            manifest["protocol"] = protocol
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
            if gold := compute_priority_gold_metrics(state.prioritized_requirements, ground_truth):
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
        model_slug = re.sub(r"[^A-Za-z0-9._-]", "-", config.model)
        run_id = (
            f"{strategy.name}_{model_slug}_{config.lang}_n{len(requirements)}_{int(time.time())}"
        )
        run_path = self._out / run_id
        run_path.mkdir(parents=True, exist_ok=True)
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

        manifest = self._build_manifest(
            config,
            run_id,
            len(requirements),
            elapsed,
            trace_path,
            state.metrics.get("agent_runtime"),
            state.metrics.get("protocol"),
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
            self._include_ground_truth,
        )
        failed_path = write_failed_csv(
            backlog.failed,
            run_path / FAILED_FILENAME,
            self._backlog_delimiter,
            self._include_ground_truth,
        )
        actions_path = write_episodes_jsonl(self._episodes_of(state), run_path / ACTIONS_FILENAME)
        results = {
            "run_id": run_id,
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
        )
