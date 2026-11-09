"""Dataset and language selection through the runner, the backlog and the CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cli.main import app
from config.settings import settings
from domain.enums import Lang, MoSCoWPriority, RequirementType
from domain.models import PipelineState, PrioritizedRequirement, Requirement
from evaluation.backlog import build_backlog, write_backlog_csv
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import OrchestrationStrategy

_GOLDS = [MoSCoWPriority.MUST_HAVE, MoSCoWPriority.SHOULD_HAVE, MoSCoWPriority.COULD_HAVE]


def _requirements() -> list[Requirement]:
    return [
        Requirement(
            id=f"r{i}",
            text=f"Requisito de teste número {i}",
            metadata={
                "label_type": "F",
                "label_category": "F",
                "ground_truth_priority": gold,
                "ground_truth_relevance": 3,
            },
        )
        for i, gold in enumerate(_GOLDS)
    ]


class EchoGoldStrategy(OrchestrationStrategy):
    """Prioritizes every item as Must: right for one item, wrong for the other two."""

    @property
    def name(self) -> str:
        return "echo"

    def execute(self, requirements, trace_writer=None) -> PipelineState:
        state = PipelineState(raw_requirements=requirements)
        state.prioritized_requirements = [
            PrioritizedRequirement(
                id=r.id,
                text=r.text,
                requirement_type=RequirementType.FUNCTIONAL,
                confidence=0.9,
                priority=MoSCoWPriority.MUST_HAVE,
                priority_score=1.0,
                priority_rank=i + 1,
                metadata=r.metadata,
            )
            for i, r in enumerate(requirements)
        ]
        return state


def _run(tmp_path: Path, **config_overrides):
    adapter = MagicMock()
    adapter.load.return_value = adapter.load_sample.return_value = _requirements()
    with patch("experiments.runner.build_adapter", return_value=adapter) as build:
        runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
        config = RunConfig(
            strategy_name="echo",
            model="fake/model",
            dataset_path="x.csv",
            n_samples=3,
            **config_overrides,
        )
        return runner.execute(EchoGoldStrategy(), config), build


class TestRunnerUsesTheRegistry:
    def test_o_adapter_vem_do_registro_com_nome_caminho_e_idioma_do_dado(self, tmp_path):
        _, build = _run(tmp_path, dataset_name="priorespa", data_lang="pt", lang="es")

        build.assert_called_once_with("priorespa", "x.csv", Lang.PT)

    def test_manifesto_e_resultados_registram_a_celula_de_idioma(self, tmp_path):
        result, _ = _run(tmp_path, dataset_name="priorespa", data_lang="es", lang="pt")

        results = json.loads(next(tmp_path.glob("**/results.json")).read_text(encoding="utf-8"))
        for record in (result.manifest, results["config"]):
            assert (record["dataset_name"], record["data_lang"]) == ("priorespa", "es")
        assert result.manifest["lang"] == "pt"

    def test_padrao_continua_sendo_promise_em_portugues(self, tmp_path):
        result, build = _run(tmp_path)

        assert (result.config.dataset_name, result.config.data_lang) == ("promise", "pt")
        build.assert_called_once_with("promise", "x.csv", Lang.PT)

    def test_idioma_do_dado_nao_se_confunde_com_o_do_prompt(self):
        config = RunConfig("s", "m", "x.csv", lang="pt", dataset_name="priorespa")

        assert (config.lang, config.data_lang) == ("pt", "es")


class TestPriorityMetricsInTheRun:
    def test_prioridade_contra_o_gabarito_entra_em_metrics(self, tmp_path):
        result, _ = _run(tmp_path, dataset_name="priorespa")

        gold = result.metrics["prioritization"]
        assert gold["accuracy"] == round(1 / 3, 4)
        assert (gold["n_total"], gold["n_evaluated"], gold["coverage"]) == (3, 3, 1.0)

    def test_dataset_sem_gabarito_de_prioridade_nao_ganha_a_chave(self, tmp_path):
        adapter = MagicMock()
        plain = [
            Requirement(
                id="a",
                text="Requisito sem gabarito de prioridade",
                metadata={"label_type": "F", "label_category": "F"},
            )
        ]
        adapter.load.return_value = adapter.load_sample.return_value = plain
        with patch("experiments.runner.build_adapter", return_value=adapter):
            runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
            result = runner.execute(
                EchoGoldStrategy(), RunConfig("echo", "fake/model", "x.csv", n_samples=1)
            )

        assert "prioritization" not in result.metrics


class TestBacklogGoldPriority:
    @staticmethod
    def _backlog():
        requirements = _requirements()
        state = EchoGoldStrategy().execute(requirements)
        return build_backlog(state, "run-1")

    def test_linha_traz_gabarito_e_acerto_da_prioridade(self):
        rows = {row.requirement_id: row for row in self._backlog().rows}

        assert (rows["r0"].reference_priority, rows["r0"].priority_correct) == ("M", True)
        assert (rows["r1"].reference_priority, rows["r1"].priority_correct) == ("S", False)

    def test_csv_ganha_as_colunas_de_gabarito_de_prioridade(self, tmp_path):
        path = write_backlog_csv(self._backlog().rows, tmp_path / "backlog.csv")

        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=";"))

        assert {"reference_priority", "priority_correct"} <= set(rows[0])
        assert {row["reference_priority"] for row in rows} == {"M", "S", "C"}

    def test_sem_gabarito_as_colunas_saem_do_csv(self, tmp_path):
        path = write_backlog_csv(
            self._backlog().rows, tmp_path / "backlog.csv", include_reference=False
        )

        header = path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert "reference_priority" not in header and "priority_correct" not in header


class TestCli:
    @staticmethod
    def _invoke(*args: str):
        with (
            patch("cli.main.PipelineStrategy"),
            patch("cli.main.ExperimentRunner") as runner,
        ):
            runner.return_value.execute.return_value = MagicMock(
                metrics={}, backlog_path="b.csv", failed_path=None, actions_path=None
            )
            result = CliRunner().invoke(
                app, ["run", "--strategy", "pipeline", "--no-preflight", *args]
            )
        return result, runner

    def test_priorespa_usa_o_arquivo_e_o_idioma_originais(self):
        result, runner = self._invoke("--dataset-name", "priorespa", "--lang", "es", "--n", "1")

        assert result.exit_code == 0, result.output
        config = runner.return_value.execute.call_args.args[1]
        assert config.dataset_name == "priorespa" and config.data_lang == "es"
        assert config.dataset_path == settings.priorespa_dataset_path and config.lang == "es"

    def test_dado_em_portugues_aponta_para_a_traducao(self):
        result, runner = self._invoke(
            "--dataset-name", "priorespa", "--data-lang", "pt", "--lang", "pt"
        )

        config = runner.return_value.execute.call_args.args[1]
        assert config.data_lang == "pt"
        assert config.dataset_path == settings.priorespa_pt_dataset_path

    def test_prompt_em_portugues_sobre_dado_em_espanhol(self):
        _, runner = self._invoke("--dataset-name", "priorespa", "--lang", "pt")

        config = runner.return_value.execute.call_args.args[1]
        assert (config.lang, config.data_lang) == ("pt", "es")

    def test_dataset_desconhecido_e_recusado(self):
        result, _ = self._invoke("--dataset-name", "nope")

        assert result.exit_code != 0
