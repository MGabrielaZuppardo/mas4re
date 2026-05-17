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
from experiments.strategy import OrchestrationStrategy

logger = logging.getLogger(__name__)


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
    n_samples: int | None = None
    seed: int = 42
    temperature: float = 0.0
    prompt_version: str = "v1"


@dataclass
class RunResult:
    config: RunConfig
    state: PipelineState
    elapsed_seconds: float
    manifest: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Runs a strategy under a frozen config and writes a reproducible
    manifest. Same runner for every architecture so SQ2 comparisons are
    controlled.
    """

    def __init__(self, out_dir: str = "experiments/results") -> None:
        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)

    def _build_manifest(self, config: RunConfig, n_loaded: int, elapsed: float) -> dict[str, Any]:
        return {
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "strategy": config.strategy_name,
            "model": config.model,
            "seed": config.seed,
            "temperature": config.temperature,
            "prompt_version": config.prompt_version,
            "dataset": config.dataset_path,
            "dataset_n": n_loaded,
            "dataset_md5": _file_md5(Path(config.dataset_path)),
            "langgraph_version": _langgraph_version(),
            "elapsed_seconds": round(elapsed, 3),
        }

    def _compute_metrics(self, state: PipelineState) -> dict[str, Any]:
        """Compute quality metrics from the processed state.

        Classification metrics use PROMISE gold labels (raw_requirements
        metadata). Prioritization has no gold labels in PROMISE, so we
        report the MoSCoW distribution only. Predictions come from
        prioritized_requirements when available (covers both pipeline
        and baseline, since PrioritizedRequirement extends
        ClassifiedRequirement), falling back to classified_requirements.
        """
        ground_truth = state.raw_requirements
        # PrioritizedRequirement extends ClassifiedRequirement; the metrics
        # functions only read fields present on the base class. list is
        # invariant for mypy, so an explicit cast is required here.
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
        return metrics

    def execute(self, strategy: OrchestrationStrategy, config: RunConfig) -> RunResult:
        adapter = PromiseAdapter(path=config.dataset_path)
        requirements = (
            adapter.load_sample(config.n_samples, seed=config.seed)
            if config.n_samples
            else adapter.load()
        )
        logger.info(
            "Run start | strategy=%s | model=%s | n=%d | seed=%d",
            strategy.name,
            config.model,
            len(requirements),
            config.seed,
        )

        start = time.perf_counter()
        state = strategy.execute(requirements)
        elapsed = time.perf_counter() - start

        manifest = self._build_manifest(config, len(requirements), elapsed)

        # Filesystem-safe slug: model ids carry "/", ":" and "+"
        # (e.g. "ollama/qwen2.5:7b") which are invalid in Windows paths.
        model_slug = re.sub(r"[^A-Za-z0-9._-]", "-", config.model)
        run_id = f"{strategy.name}_{model_slug}_n{len(requirements)}_{int(time.time())}"
        run_path = self._out / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        metrics = self._compute_metrics(state)
        predictions = [r.model_dump(mode="json") for r in state.prioritized_requirements] or [
            r.model_dump(mode="json") for r in state.classified_requirements
        ]
        results = {
            "config": {
                "strategy": config.strategy_name,
                "model": config.model,
                "seed": config.seed,
                "n": len(requirements),
            },
            "metrics": metrics,
            "n_predictions": len(predictions),
            "predictions": predictions,
        }
        (run_path / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

        logger.info(
            "Run done | strategy=%s | elapsed=%.2fs | metrics=%s | dir=%s",
            strategy.name,
            elapsed,
            metrics.get("classification", {}),
            run_path,
        )
        return RunResult(
            config=config,
            state=state,
            elapsed_seconds=elapsed,
            manifest=manifest,
            metrics=metrics,
        )
