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

from datasets.promise import PromiseAdapter
from domain.models import ClassifiedRequirement, PipelineState
from evaluation.metrics.classification import (
    compute_classification_metrics,
    compute_subcategory_metrics,
)
from evaluation.metrics.prioritization import compute_moscow_distribution
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
    """Frozen parameters of one experimental run (SQ2 reproducibility)."""

    strategy_name: str
    model: str
    dataset_path: str
    lang: str = "pt"
    n_samples: int | None = None
    seed: int = 42
    temperature: float = 0.0
    prompt_version: str = "v1"


@dataclass
class RunResult:
    config: RunConfig
    state: PipelineState
    elapsed_seconds: float
    run_id: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Runs a strategy under a frozen config and writes a reproducible
    manifest. Same runner for every architecture so SQ2 comparisons are
    controlled.

    Each run produces three artifacts in experiments/results/{run_id}/:
        manifest.json  — frozen parameters + git commit + dataset hash
        results.json   — quality metrics + full predictions
        trace.jsonl    — per-requirement latency + parse outcome (TraceWriter)
    """

    def __init__(self, out_dir: str = "experiments/results") -> None:
        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)

    def _build_manifest(
        self,
        config: RunConfig,
        run_id: str,
        n_loaded: int,
        elapsed: float,
        trace_path: Path,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "strategy": config.strategy_name,
            "model": config.model,
            "lang": config.lang,
            "seed": config.seed,
            "temperature": config.temperature,
            "prompt_version": config.prompt_version,
            "dataset": config.dataset_path,
            "dataset_n": n_loaded,
            "dataset_md5": _file_md5(Path(config.dataset_path)),
            "langgraph_version": _langgraph_version(),
            "elapsed_seconds": round(elapsed, 3),
            "trace_path": str(trace_path),
        }

    def _compute_metrics(self, state: PipelineState) -> dict[str, Any]:
        ground_truth = state.raw_requirements
        classified = cast(
            "list[ClassifiedRequirement]",
            state.prioritized_requirements or state.classified_requirements,
        )

        # Starts from state.metrics (not {}): cross_check_node (ADR-003) writes
        # inter_agent_conflicts there when the pipeline architecture is used,
        # and it must survive into the persisted results.json, not be
        # silently discarded by this dict being rebuilt from scratch.
        metrics: dict[str, Any] = dict(state.metrics)
        if classified:
            metrics["classification"] = compute_classification_metrics(classified, ground_truth)
            metrics["subcategory"] = compute_subcategory_metrics(classified, ground_truth)
        if state.prioritized_requirements:
            metrics["moscow_distribution"] = compute_moscow_distribution(
                state.prioritized_requirements
            )
        return metrics

    def execute(self, strategy: OrchestrationStrategy, config: RunConfig) -> RunResult:
        adapter = PromiseAdapter(path=config.dataset_path)
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
        trace_path = _TRACE_DIR / f"{run_id}.jsonl"

        logger.info(
            "Run start | run_id=%s | strategy=%s | model=%s | lang=%s | n=%d",
            run_id,
            strategy.name,
            config.model,
            config.lang,
            len(requirements),
        )

        start = time.perf_counter()
        with TraceWriter(output_dir=_TRACE_DIR, run_id=run_id) as tw:
            state = strategy.execute(requirements, trace_writer=tw)
        elapsed = time.perf_counter() - start

        manifest = self._build_manifest(config, run_id, len(requirements), elapsed, trace_path)
        (run_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        metrics = self._compute_metrics(state)
        predictions = [r.model_dump(mode="json") for r in state.prioritized_requirements] or [
            r.model_dump(mode="json") for r in state.classified_requirements
        ]
        failure_records = [r.model_dump(mode="json") for r in state.failure_records]
        results = {
            "run_id": run_id,
            "config": {
                "strategy": config.strategy_name,
                "model": config.model,
                "lang": config.lang,
                "seed": config.seed,
                "n": len(requirements),
            },
            "metrics": metrics,
            "n_predictions": len(predictions),
            "predictions": predictions,
            "errors": state.errors,
            "n_failure_records": len(failure_records),
            "failure_records": failure_records,
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
        )
