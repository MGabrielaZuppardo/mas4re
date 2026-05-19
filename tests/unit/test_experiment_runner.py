"""Unit tests for the ExperimentRunner and strategy contract."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain.enums import MoSCoWPriority, RequirementType
from domain.models import PipelineState, PrioritizedRequirement, Requirement
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import OrchestrationStrategy


class FakeStrategy(OrchestrationStrategy):
    @property
    def name(self) -> str:
        return "fake"

    def execute(self, requirements: list[Requirement], trace_writer=None) -> PipelineState:
        state = PipelineState(raw_requirements=requirements)
        state.prioritized_requirements = [
            PrioritizedRequirement(
                text=r.text,
                requirement_type=RequirementType.FUNCTIONAL,
                priority=MoSCoWPriority.MUST_HAVE,
                priority_score=1.0,
                priority_rank=i + 1,
                priority_justification="fake",
            )
            for i, r in enumerate(requirements)
        ]
        return state


@pytest.fixture
def fake_requirements() -> list[Requirement]:
    return [
        Requirement(id=f"r{i}", text=f"Requisito de teste número {i} do sistema.") for i in range(5)
    ]


def _patch_adapter(reqs: list[Requirement]):
    fake_adapter = MagicMock()
    fake_adapter.load.return_value = reqs
    fake_adapter.load_sample.return_value = reqs
    return patch("experiments.runner.PromiseAdapter", return_value=fake_adapter)


def test_runner_executes_strategy(tmp_path: Path, fake_requirements) -> None:
    with _patch_adapter(fake_requirements):
        runner = ExperimentRunner(out_dir=str(tmp_path))
        config = RunConfig(
            strategy_name="fake",
            model="fake/model",
            dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
            n_samples=5,
        )
        result = runner.execute(FakeStrategy(), config)

    assert len(result.state.prioritized_requirements) == 5
    assert result.manifest["strategy"] == "fake"
    assert result.manifest["seed"] == 42
    assert result.manifest["dataset_n"] == 5
    assert "git_commit" in result.manifest


def test_runner_writes_manifest(tmp_path: Path, fake_requirements) -> None:
    with _patch_adapter(fake_requirements):
        runner = ExperimentRunner(out_dir=str(tmp_path))
        config = RunConfig(
            strategy_name="fake",
            model="fake/model",
            dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
            n_samples=5,
        )
        runner.execute(FakeStrategy(), config)

    manifests = list(tmp_path.glob("**/manifest.json"))
    assert len(manifests) == 1
    data = json.loads(manifests[0].read_text())
    assert data["strategy"] == "fake"
    assert data["prompt_version"] == "v1"
    assert "timestamp_utc" in data


def test_manifest_deterministic_fields(tmp_path: Path, fake_requirements) -> None:
    """Same config -> same deterministic fields (seed, model, strategy)."""
    with _patch_adapter(fake_requirements):
        runner = ExperimentRunner(out_dir=str(tmp_path))
        config = RunConfig(
            strategy_name="fake",
            model="fake/model",
            dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
            n_samples=5,
            seed=42,
        )
        r1 = runner.execute(FakeStrategy(), config)
        r2 = runner.execute(FakeStrategy(), config)

    for key in ("strategy", "model", "seed", "temperature", "prompt_version"):
        assert r1.manifest[key] == r2.manifest[key]


def test_runner_persists_results_json(tmp_path: Path, fake_requirements) -> None:
    with _patch_adapter(fake_requirements):
        runner = ExperimentRunner(out_dir=str(tmp_path))
        config = RunConfig(
            strategy_name="fake",
            model="fake/model",
            dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
            n_samples=5,
        )
        result = runner.execute(FakeStrategy(), config)

    results_files = list(tmp_path.glob("**/results.json"))
    assert len(results_files) == 1
    data = json.loads(results_files[0].read_text())
    assert data["n_predictions"] == 5
    assert data["config"]["strategy"] == "fake"
    assert "metrics" in data
    assert "moscow_distribution" in result.metrics


def test_runner_metrics_moscow_distribution_sums_to_one(tmp_path: Path, fake_requirements) -> None:
    with _patch_adapter(fake_requirements):
        runner = ExperimentRunner(out_dir=str(tmp_path))
        config = RunConfig(
            strategy_name="fake",
            model="fake/model",
            dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
            n_samples=5,
        )
        result = runner.execute(FakeStrategy(), config)

    dist = result.metrics["moscow_distribution"]
    assert abs(sum(dist.values()) - 1.0) < 1e-6
