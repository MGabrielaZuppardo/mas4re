"""Unit tests for the ExperimentRunner and strategy contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain.enums import InformationRegime, MoSCoWPriority, RequirementType
from domain.models import (
    ClassificationOutput,
    ClassifiedRequirement,
    PipelineState,
    PrioritizationOutput,
    PrioritizedRequirement,
    Requirement,
)
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
        runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
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
        runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
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
        runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
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
        runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
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
        runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
        config = RunConfig(
            strategy_name="fake",
            model="fake/model",
            dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
            n_samples=5,
        )
        result = runner.execute(FakeStrategy(), config)

    dist = result.metrics["moscow_distribution"]
    assert abs(sum(dist.values()) - 1.0) < 1e-6


class FailureDetectingStrategy(OrchestrationStrategy):
    """Fake strategy that populates state.metrics["failure_detections"]
    the same way agents/base.py::_persist_failure_records does, to test
    that ExperimentRunner._compute_metrics propagates it into results.json."""

    @property
    def name(self) -> str:
        return "fake_failure"

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
        state.metrics["failure_detections"] = [
            {"requirement_id": requirements[0].id, "stage": "classify", "mode": "schema_invalid"}
        ]
        return state


def test_runner_propagates_failure_detections_to_metrics(tmp_path: Path, fake_requirements) -> None:
    with _patch_adapter(fake_requirements):
        runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
        config = RunConfig(
            strategy_name="fake_failure",
            model="fake/model",
            dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
            n_samples=5,
        )
        result = runner.execute(FailureDetectingStrategy(), config)

    assert result.metrics["failure_detections"] == [
        {"requirement_id": fake_requirements[0].id, "stage": "classify", "mode": "schema_invalid"}
    ]
    results_files = list(tmp_path.glob("**/results.json"))
    data = json.loads(results_files[0].read_text())
    assert data["metrics"]["failure_detections"]


class TestInformationRegime:
    def test_run_config_e_zero_shot_por_padrao(self) -> None:
        config = RunConfig(strategy_name="fake", model="m", dataset_path="d.csv")

        assert config.information_regime is InformationRegime.ZERO_SHOT

    def test_run_config_aceita_o_valor_em_texto(self) -> None:
        config = RunConfig(
            strategy_name="fake",
            model="m",
            dataset_path="d.csv",
            information_regime="hybrid_exploratory",  # type: ignore[arg-type]
        )

        assert config.information_regime is InformationRegime.HYBRID_EXPLORATORY

    def test_run_config_recusa_regime_desconhecido(self) -> None:
        with pytest.raises(ValueError):
            RunConfig(
                strategy_name="fake",
                model="m",
                dataset_path="d.csv",
                information_regime="few_shot",  # type: ignore[arg-type]
            )

    def test_manifest_e_results_registram_o_regime(self, tmp_path: Path, fake_requirements) -> None:
        with _patch_adapter(fake_requirements):
            runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
            config = RunConfig(
                strategy_name="fake",
                model="fake/model",
                dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
                n_samples=5,
                information_regime=InformationRegime.HYBRID_EXPLORATORY,
            )
            result = runner.execute(FakeStrategy(), config)

        results_file = next(tmp_path.glob("**/results.json"))
        manifest_file = next(tmp_path.glob("**/manifest.json"))
        assert result.manifest["information_regime"] == "hybrid_exploratory"
        assert json.loads(manifest_file.read_text())["information_regime"] == "hybrid_exploratory"
        assert json.loads(results_file.read_text())["config"]["information_regime"] == (
            "hybrid_exploratory"
        )


def _labeled_requirements() -> list[Requirement]:
    return [
        Requirement(
            id=f"r{i}",
            text=f"Requisito rotulado número {i}.",
            metadata={"label_type": "F", "label_category": "F"},
        )
        for i in range(4)
    ]


class ProcessingStrategy(OrchestrationStrategy):
    """Classifies and prioritizes the requirements, keeping their ids. With `lose_last`,
    the last one is lost after a fatal detection, as when an agent gives up."""

    def __init__(self, lose_last: bool = False) -> None:
        self._lose_last = lose_last

    @property
    def name(self) -> str:
        return "processing"

    def execute(self, requirements: list[Requirement], trace_writer=None) -> PipelineState:
        state = PipelineState(raw_requirements=requirements)
        processed = requirements[:-1] if self._lose_last else requirements
        state.classified_requirements = [
            ClassifiedRequirement.from_requirement(
                r,
                ClassificationOutput(
                    requirement_id=r.id,
                    requirement_type=RequirementType.FUNCTIONAL,
                    confidence=0.9,
                    justification="ok",
                ),
            )
            for r in processed
        ]
        state.prioritized_requirements = [
            PrioritizedRequirement.from_classified(
                c,
                PrioritizationOutput(
                    requirement_id=c.id,
                    priority=MoSCoWPriority.MUST_HAVE,
                    priority_score=1.0,
                    priority_rank=i + 1,
                    justification="ok",
                ),
            )
            for i, c in enumerate(state.classified_requirements)
        ]
        if self._lose_last:
            state.metrics["failure_detections"] = [
                {
                    "requirement_id": requirements[-1].id,
                    "stage": "classify",
                    "mode": "schema_invalid",
                    "severity": "fatal",
                    "evidence": "parser fell back",
                }
            ]
        return state


def _execute(tmp_path: Path, strategy: OrchestrationStrategy, **runner_options):
    with _patch_adapter(_labeled_requirements()):
        runner = ExperimentRunner(
            out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"), **runner_options
        )
        config = RunConfig(
            strategy_name="processing",
            model="fake/model",
            dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
            n_samples=4,
        )
        return runner.execute(strategy, config)


def _read_csv(path: Path, delimiter: str = ";") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


class TestBacklogArtifacts:
    def test_grava_o_backlog_e_o_arquivo_de_falhas_separados(self, tmp_path: Path) -> None:
        result = _execute(tmp_path, ProcessingStrategy(lose_last=True))

        backlog = _read_csv(result.backlog_path)
        failed = _read_csv(result.failed_path)
        assert [row["requirement_id"] for row in backlog] == ["r0", "r1", "r2"]
        assert [row["requirement_id"] for row in failed] == ["r3"]
        assert failed[0]["failure_evidence"] == "parser fell back"

    def test_results_json_aponta_os_arquivos_e_conta_as_falhas(self, tmp_path: Path) -> None:
        result = _execute(tmp_path, ProcessingStrategy(lose_last=True))

        data = json.loads(next(tmp_path.glob("**/results.json")).read_text())

        assert data["n_failed"] == 1
        assert data["backlog_path"] == str(result.backlog_path)
        assert data["failed_path"] == str(result.failed_path)

    def test_sem_falhas_nao_cria_o_arquivo_de_falhas(self, tmp_path: Path) -> None:
        result = _execute(tmp_path, ProcessingStrategy(lose_last=False))

        data = json.loads(next(tmp_path.glob("**/results.json")).read_text())

        assert result.failed_path is None and data["failed_path"] is None
        assert data["n_failed"] == 0
        assert not list(tmp_path.glob("**/failed.csv"))
        assert len(_read_csv(result.backlog_path)) == 4

    def test_separador_e_gabarito_configuraveis(self, tmp_path: Path) -> None:
        result = _execute(
            tmp_path,
            ProcessingStrategy(lose_last=True),
            backlog_delimiter=",",
            include_ground_truth=False,
        )

        backlog = _read_csv(result.backlog_path, delimiter=",")
        failed = _read_csv(result.failed_path, delimiter=",")
        assert "gold_type" not in backlog[0] and "gold_type" not in failed[0]
        assert backlog[0]["requirement_type"] == "F"
