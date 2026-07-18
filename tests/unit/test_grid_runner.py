"""Unit tests for grid-runner changes (PR #24).

Tests cover:
- RunConfig.lang field
- ExperimentRunner wires TraceWriter (trace.jsonl is created)
- strategy.execute() receives and injects trace_writer into agents
- run_id present in RunResult and artifacts
- grid script dry-run produces correct condition count

All tests use mocked LLM — no Ollama required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from domain.models import PipelineState, Requirement
from evaluation.trace_writer import TraceWriter
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import BaselineStrategy, PipelineStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reqs(n: int = 3) -> list[Requirement]:
    return [
        Requirement(
            text=f"The system shall do thing {i}.",
            source="test",
            metadata={"label_type": "F"},
        )
        for i in range(n)
    ]


def _baseline_llm_response() -> MagicMock:
    mock = MagicMock()
    mock.content = json.dumps(
        {
            "requirement_type": "F",
            "nfr_category": None,
            "confidence": 0.9,
            "classification_justification": "ok",
            "priority": "M",
            "priority_score": 0.9,
            "priority_rank": 1,
            "priority_justification": "critical",
        }
    )
    return mock


def _classify_llm_response() -> MagicMock:
    mock = MagicMock()
    mock.content = json.dumps(
        {
            "requirement_type": "F",
            "nfr_category": None,
            "confidence": 0.9,
            "justification": "ok",
        }
    )
    return mock


def _prioritize_llm_response() -> MagicMock:
    mock = MagicMock()
    mock.content = json.dumps(
        {
            "priority": "M",
            "priority_score": 0.9,
            "priority_rank": 1,
            "justification": "critical",
        }
    )
    return mock


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_lang_default_is_pt(self) -> None:
        cfg = RunConfig(strategy_name="baseline", model="m", dataset_path="d")
        assert cfg.lang == "pt"

    def test_lang_can_be_set_to_en(self) -> None:
        cfg = RunConfig(strategy_name="baseline", model="m", dataset_path="d", lang="en")
        assert cfg.lang == "en"

    def test_run_id_not_in_config(self) -> None:
        cfg = RunConfig(strategy_name="baseline", model="m", dataset_path="d")
        assert not hasattr(cfg, "run_id")


# ---------------------------------------------------------------------------
# BaselineStrategy.execute() — trace_writer injection
# ---------------------------------------------------------------------------


class TestBaselineStrategyTrace:
    def test_execute_without_trace_writer(self) -> None:
        with patch("agents.baseline.build_llm") as mock_build:
            mock_build.return_value = MagicMock(
                invoke=MagicMock(return_value=_baseline_llm_response())
            )
            strat = BaselineStrategy(model="stub")
            reqs = _make_reqs(2)
            state = strat.execute(reqs, trace_writer=None)
        assert isinstance(state, PipelineState)

    def test_execute_injects_trace_writer(self, tmp_path: Path) -> None:
        with patch("agents.baseline.build_llm") as mock_build:
            mock_build.return_value = MagicMock(
                invoke=MagicMock(return_value=_baseline_llm_response())
            )
            strat = BaselineStrategy(model="stub")
            reqs = _make_reqs(2)
            with TraceWriter(output_dir=tmp_path, run_id="strat-test") as tw:
                strat.execute(reqs, trace_writer=tw)

        lines = (tmp_path / "strat-test.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert all(json.loads(raw)["stage"] == "baseline" for raw in lines)

    def test_trace_writer_cleared_after_execute(self, tmp_path: Path) -> None:
        """Agent _trace should not leak across runs."""
        with patch("agents.baseline.build_llm") as mock_build:
            mock_build.return_value = MagicMock(
                invoke=MagicMock(return_value=_baseline_llm_response())
            )
            strat = BaselineStrategy(model="stub")
            with TraceWriter(output_dir=tmp_path, run_id="run-a") as tw:
                strat.execute(_make_reqs(1), trace_writer=tw)
            # Second execute without trace_writer — agent._trace must be None
            strat.execute(_make_reqs(1), trace_writer=None)
            assert strat._agent._trace is None


# ---------------------------------------------------------------------------
# PipelineStrategy.execute() — trace_writer injection
# ---------------------------------------------------------------------------


class TestPipelineStrategyTrace:
    def test_execute_injects_trace_into_both_agents(self, tmp_path: Path) -> None:
        with (
            patch("agents.classifier.build_llm") as mock_cls,
            patch("agents.prioritizer.build_llm") as mock_pri,
        ):
            mock_cls.return_value = MagicMock(
                invoke=MagicMock(return_value=_classify_llm_response())
            )
            mock_pri.return_value = MagicMock(
                invoke=MagicMock(return_value=_prioritize_llm_response())
            )
            strat = PipelineStrategy(
                classifier_model="stub-cls",
                prioritizer_model="stub-pri",
            )
            with TraceWriter(output_dir=tmp_path, run_id="pipe-test") as tw:
                strat.execute(_make_reqs(2), trace_writer=tw)

        lines = (tmp_path / "pipe-test.jsonl").read_text().splitlines()
        stages = {json.loads(raw)["stage"] for raw in lines}
        # Both classify and prioritize stages must appear
        assert "classify" in stages
        assert "prioritize" in stages

    def test_classifier_and_prioritizer_refs_are_stored(self) -> None:
        with (
            patch("agents.classifier.build_llm"),
            patch("agents.prioritizer.build_llm"),
        ):
            strat = PipelineStrategy(
                classifier_model="stub-cls",
                prioritizer_model="stub-pri",
            )
        assert hasattr(strat, "_classifier")
        assert hasattr(strat, "_prioritizer")


# ---------------------------------------------------------------------------
# ExperimentRunner — TraceWriter wiring + run_id in artifacts
# ---------------------------------------------------------------------------


class TestExperimentRunnerTrace:
    def _run_baseline(self, tmp_path: Path, n: int = 2) -> tuple:
        results_dir = tmp_path / "results"
        traces_dir = tmp_path / "traces"

        reqs = _make_reqs(n)
        with patch("agents.baseline.build_llm") as mock_build:
            mock_build.return_value = MagicMock(
                invoke=MagicMock(return_value=_baseline_llm_response())
            )
            with patch("experiments.runner.PromiseAdapter") as mock_adapter:
                mock_adapter.return_value.load_sample.return_value = reqs
                mock_adapter.return_value.load.return_value = reqs

                strat = BaselineStrategy(model="stub")
                cfg = RunConfig(
                    strategy_name="baseline",
                    model="stub",
                    dataset_path="fake.csv",
                    lang="pt",
                    n_samples=n,
                )
                runner = ExperimentRunner(out_dir=str(results_dir), trace_dir=str(traces_dir))
                result = runner.execute(strat, cfg)
        return result, results_dir, traces_dir

    def test_run_result_has_run_id(self, tmp_path: Path) -> None:
        result, _, _ = self._run_baseline(tmp_path)
        assert result.run_id != ""
        assert "baseline" in result.run_id

    def test_manifest_contains_run_id(self, tmp_path: Path) -> None:
        result, results_dir, _ = self._run_baseline(tmp_path)
        manifest_path = results_dir / result.run_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["run_id"] == result.run_id

    def test_manifest_contains_lang(self, tmp_path: Path) -> None:
        result, results_dir, _ = self._run_baseline(tmp_path)
        manifest = json.loads((results_dir / result.run_id / "manifest.json").read_text())
        assert manifest["lang"] == "pt"

    def test_results_json_contains_run_id(self, tmp_path: Path) -> None:
        result, results_dir, _ = self._run_baseline(tmp_path)
        results = json.loads((results_dir / result.run_id / "results.json").read_text())
        assert results["run_id"] == result.run_id

    def test_trace_jsonl_created(self, tmp_path: Path) -> None:
        result, _, traces_dir = self._run_baseline(tmp_path)
        trace_file = traces_dir / f"{result.run_id}.jsonl"
        assert trace_file.exists()
        lines = trace_file.read_text().splitlines()
        assert len(lines) == 2  # one per requirement

    def test_trace_run_id_matches_manifest(self, tmp_path: Path) -> None:
        result, _, traces_dir = self._run_baseline(tmp_path)
        trace_file = traces_dir / f"{result.run_id}.jsonl"
        for line in trace_file.read_text().splitlines():
            assert json.loads(line)["run_id"] == result.run_id


# ---------------------------------------------------------------------------
# Grid script — dry-run
# ---------------------------------------------------------------------------


class TestGridScript:
    def test_dry_run_produces_18_conditions(self, capsys) -> None:
        from scripts.run_grid import _build_conditions

        conditions = _build_conditions()
        assert len(conditions) == 18

    def test_conditions_cover_all_models_langs_strategies(self) -> None:
        from scripts.run_grid import LANGUAGES, MODELS, _build_conditions

        conditions = _build_conditions()
        for model in MODELS:
            for lang in LANGUAGES:
                for strategy in ("baseline", "pipeline", "two_call_baseline"):
                    match = [
                        c
                        for c in conditions
                        if c.model == model and c.lang == lang and c.strategy == strategy
                    ]
                    assert len(match) == 1, f"Missing: {model} {lang} {strategy}"

    def test_baseline_strategy_uses_same_model(self) -> None:
        from scripts.run_grid import GridCondition, _build_config

        cond = GridCondition(model="ollama/qwen2.5:7b", lang="pt", strategy="baseline")
        cfg = _build_config(cond, n=5)
        assert cfg.model == "ollama/qwen2.5:7b"
        assert cfg.lang == "pt"
        assert cfg.n_samples == 5

    def test_pipeline_config_model_label(self) -> None:
        from scripts.run_grid import GridCondition, _build_config

        cond = GridCondition(model="ollama/qwen2.5:7b", lang="en", strategy="pipeline")
        cfg = _build_config(cond, n=None)
        assert cfg.model == "ollama/qwen2.5:7b+ollama/qwen2.5:7b"
        assert cfg.lang == "en"

    def test_two_call_baseline_config_model_label(self) -> None:
        from scripts.run_grid import GridCondition, _build_config

        cond = GridCondition(model="ollama/qwen2.5:7b", lang="en", strategy="two_call_baseline")
        cfg = _build_config(cond, n=None)
        # Unlike "pipeline", two_call_baseline is a single agent making two
        # calls, not two agent slots — model_label must not get a "+model"
        # suffix (see scripts/run_grid.py::_build_config).
        assert cfg.model == "ollama/qwen2.5:7b"
        assert cfg.lang == "en"

    def test_build_strategy_dispatches_two_call_baseline(self) -> None:
        from experiments.strategy import TwoCallBaselineStrategy
        from scripts.run_grid import GridCondition, _build_strategy

        cond = GridCondition(model="ollama/qwen2.5:7b", lang="pt", strategy="two_call_baseline")
        strategy = _build_strategy(cond)
        assert isinstance(strategy, TwoCallBaselineStrategy)
