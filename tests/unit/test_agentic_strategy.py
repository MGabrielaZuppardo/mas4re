"""The autonomous agents inside the experiment machinery: strategy, graph, runner, CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer.main
from typer.testing import CliRunner

from cli.main import app
from domain.agent_actions import AgentEpisode
from domain.enums import CoordinationProtocol, RequirementType
from domain.messages import ClassificationReport, Performative
from domain.models import PipelineState, Requirement
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import AgenticPipelineStrategy, OrchestrationStrategy
from tests.helpers.scripted_llm import ScriptedChatModel, tool_call

_TEXT = "O sistema deve responder em 200ms"


def _requirements() -> list[Requirement]:
    return [
        Requirement(
            id="r1", text=_TEXT, source="P1", metadata={"label_type": "NF", "label_category": "PE"}
        )
    ]


def _classify(category: str = "PE"):
    return tool_call(
        "submit_classification",
        requirement_type="NF",
        nfr_category=category,
        confidence=0.9,
        justification="tempo de resposta",
    )


def _prioritize(priority: str = "M"):
    return tool_call("submit_priority", priority=priority, justification="crítico")


def _refuse_request():
    return tool_call("reply_to_request", decision="refuse", reason="mantenho")


def _strategy(
    classifier_script: list | None = None, prioritizer_script: list | None = None
) -> AgenticPipelineStrategy:
    classifier_llm = ScriptedChatModel(script=classifier_script or [_classify()])
    prioritizer_llm = ScriptedChatModel(script=prioritizer_script or [_prioritize()])
    with (
        patch("agents.agentic.classifier.build_llm", return_value=classifier_llm),
        patch("agents.agentic.prioritizer.build_llm", return_value=prioritizer_llm),
    ):
        return AgenticPipelineStrategy("ollama/clf", "ollama/pri", max_steps=4)


class TestAgenticPipelineStrategy:
    def test_e_uma_estrategia_com_nome_proprio(self):
        strategy = _strategy()

        assert isinstance(strategy, OrchestrationStrategy)
        assert strategy.name == "pipeline_agentic"

    def test_o_grafo_da_a_vez_ao_classificador_e_depois_ao_priorizador(self):
        state = _strategy().execute(_requirements())

        assert [r.id for r in state.classified_requirements] == ["r1"]
        [prioritized] = state.prioritized_requirements
        assert prioritized.priority.value == "M" and prioritized.priority_rank == 1
        assert prioritized.metadata == {"label_type": "NF", "label_category": "PE"}

    def test_a_mensagem_atravessa_o_grafo_com_o_tipo_certo(self):
        state = _strategy().execute(_requirements())

        [message] = state.messages
        assert (message.sender, message.receiver) == ("classifier", "prioritizer")
        assert message.performative is Performative.INFORM
        assert isinstance(message.content, ClassificationReport)
        assert message.content.requirement_type is RequirementType.NON_FUNCTIONAL

    def test_episodios_dos_dois_agentes_e_verificacao_cruzada_ficam_no_estado(self):
        state = _strategy().execute(_requirements())

        episodes = [AgentEpisode.model_validate(e) for e in state.metrics["agent_episodes"]]
        assert [e.agent for e in episodes] == ["classifier", "prioritizer"]
        assert set(state.metrics["agent_runtime"]) == {"classifier", "prioritizer"}
        assert "inter_agent_conflicts" in state.metrics

    def test_o_escritor_de_trace_chega_aos_dois_agentes(self):
        trace = MagicMock()

        _strategy().execute(_requirements(), trace_writer=trace)

        assert [c.args[0].stage for c in trace.write.call_args_list] == ["classify", "prioritize"]


class TestRunnerArtifacts:
    @staticmethod
    def _run(tmp_path: Path, strategy: OrchestrationStrategy):
        adapter = MagicMock()
        adapter.load.return_value = adapter.load_sample.return_value = _requirements()
        with patch("experiments.runner.build_adapter", return_value=adapter):
            runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
            config = RunConfig(
                strategy_name=strategy.name,
                model="ollama/clf+ollama/pri",
                dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
                n_samples=1,
            )
            return runner.execute(strategy, config)

    def test_grava_um_episodio_por_linha(self, tmp_path):
        result = self._run(tmp_path, _strategy())

        lines = result.actions_path.read_text(encoding="utf-8").splitlines()
        episodes = [AgentEpisode.model_validate_json(line) for line in lines]
        assert [(e.agent, e.requirement_id) for e in episodes] == [
            ("classifier", "r1"),
            ("prioritizer", "r1"),
        ]
        assert episodes[0].actions[0].name == "submit_classification"

    def test_resultado_traz_a_atividade_dos_agentes(self, tmp_path):
        result = self._run(tmp_path, _strategy())

        data = json.loads(next(tmp_path.glob("**/results.json")).read_text(encoding="utf-8"))
        activity = data["metrics"]["agent_activity"]
        assert activity["classifier"]["submitted"] == 1
        assert activity["prioritizer"]["tool_calls"] == {"submit_priority": 1}
        assert data["actions_path"] == str(result.actions_path)

    def test_manifesto_registra_orcamento_e_ferramentas(self, tmp_path):
        result = self._run(tmp_path, _strategy())

        runtime = result.manifest["agent_runtime"]
        [prioritizer_runtime] = runtime["prioritizer"]
        assert prioritizer_runtime["max_steps"] == 4
        assert "request_reclassification" not in prioritizer_runtime["tools"]
        assert prioritizer_runtime["terminal_tools"] == ["submit_priority"]
        assert len(runtime["classifier"]) == 2

    def test_manifesto_registra_o_protocolo_de_coordenacao(self, tmp_path):
        classifier_llm = ScriptedChatModel(script=[_classify("PE")])
        prioritizer_llm = ScriptedChatModel(script=[_prioritize("S")])
        with (
            patch("agents.agentic.classifier.build_llm", return_value=classifier_llm),
            patch("agents.agentic.prioritizer.build_llm", return_value=prioritizer_llm),
        ):
            strategy = AgenticPipelineStrategy(
                "ollama/clf", "ollama/pri", max_steps=4, protocol=CoordinationProtocol.PEERS
            )

        result = self._run(tmp_path, strategy)

        assert result.manifest["protocol"] == "p2"
        [prioritizer_runtime, *_] = result.manifest["agent_runtime"]["prioritizer"]
        assert "request_reclassification" in prioritizer_runtime["tools"]

    def test_resultado_traz_a_interacao_e_o_mediador_quando_o_protocolo_os_produz(self, tmp_path):
        classifier_llm = ScriptedChatModel(script=[_classify("PE")])
        prioritizer_llm = ScriptedChatModel(script=[_prioritize("M"), _refuse_request()])
        with (
            patch("agents.agentic.classifier.build_llm", return_value=classifier_llm),
            patch("agents.agentic.prioritizer.build_llm", return_value=prioritizer_llm),
        ):
            strategy = AgenticPipelineStrategy(
                "ollama/clf", "ollama/pri", max_steps=4, protocol=CoordinationProtocol.MEDIATED
            )

        result = self._run(tmp_path, strategy)

        assert result.metrics["interaction"]["requests_by_sender"] == {"mediator": 1}
        assert result.metrics["interaction"]["refuse"] == 1
        assert result.metrics["mediator"]["must_inflation"] == 1
        assert result.metrics["conformance"]["violations"] == 0
        assert result.manifest["architecture"]["protocol"] == "p3"

    def test_mensagens_trocadas_vao_para_o_messages_jsonl(self, tmp_path):
        classifier_llm = ScriptedChatModel(script=[_classify("PE")])
        prioritizer_llm = ScriptedChatModel(script=[_prioritize("M"), _refuse_request()])
        with (
            patch("agents.agentic.classifier.build_llm", return_value=classifier_llm),
            patch("agents.agentic.prioritizer.build_llm", return_value=prioritizer_llm),
        ):
            strategy = AgenticPipelineStrategy(
                "ollama/clf", "ollama/pri", max_steps=4, protocol=CoordinationProtocol.MEDIATED
            )

        result = self._run(tmp_path, strategy)

        lines = result.messages_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["performative"] for line in lines] == [
            "inform",
            "request",
            "refuse",
        ]
        data = json.loads(next(tmp_path.glob("**/results.json")).read_text(encoding="utf-8"))
        assert data["messages_path"] == str(result.messages_path)

    @staticmethod
    def _mediated_run(tmp_path):
        classifier_llm = ScriptedChatModel(script=[_classify("PE")])
        prioritizer_llm = ScriptedChatModel(script=[_prioritize("M"), _refuse_request()])
        with (
            patch("agents.agentic.classifier.build_llm", return_value=classifier_llm),
            patch("agents.agentic.prioritizer.build_llm", return_value=prioritizer_llm),
        ):
            strategy = AgenticPipelineStrategy(
                "ollama/clf", "ollama/pri", max_steps=4, protocol=CoordinationProtocol.MEDIATED
            )
        return TestRunnerArtifacts._run(tmp_path, strategy)

    def test_replay_reconstroi_as_metricas_so_dos_registros(self, tmp_path):
        from experiments.replay_metrics import replay

        result = self._mediated_run(tmp_path)

        report = replay(tmp_path / result.run_id)

        assert report["ok"], report
        assert {"agent_activity", "interaction", "conformance", "interaction_failures"} <= set(
            report["checked"]
        )

    def test_replay_acusa_metrica_adulterada(self, tmp_path):
        from experiments.replay_metrics import replay

        result = self._mediated_run(tmp_path)
        path = tmp_path / result.run_id / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["metrics"]["interaction"]["requests"] += 5
        path.write_text(json.dumps(data), encoding="utf-8")

        report = replay(tmp_path / result.run_id)

        assert not report["ok"] and report["checked"]["interaction"]["match"] is False

    def test_execucao_mediada_grava_as_decisoes_iniciais_e_finais(self, tmp_path):
        result = self._mediated_run(tmp_path)

        rows = [
            json.loads(line)
            for line in (tmp_path / result.run_id / "decisions.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]

        assert rows[0]["initial_priority"] == "M" and rows[0]["id"] == "r1"

    def test_protocolo_fixo_so_registra_o_repasse_do_classificador(self, tmp_path):
        result = self._run(tmp_path, _strategy())

        lines = result.messages_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["performative"] for line in lines] == ["inform"]

    def test_protocolo_fixo_nao_gera_interacao_nem_mediador(self, tmp_path):
        result = self._run(tmp_path, _strategy())

        assert result.metrics["interaction"]["requests"] == 0
        assert "mediator" not in result.metrics

    def test_estrategia_sem_agentes_nao_cria_o_arquivo_de_acoes(self, tmp_path):
        class Plain(OrchestrationStrategy):
            @property
            def name(self) -> str:
                return "plain"

            def execute(self, requirements, trace_writer=None) -> PipelineState:
                return PipelineState(raw_requirements=requirements)

        result = self._run(tmp_path, Plain())

        assert result.actions_path is None
        assert not list(tmp_path.glob("**/actions.jsonl"))
        assert "agent_runtime" not in result.manifest


class TestCli:
    def test_a_opcao_de_estrategia_e_o_orcamento_existem(self):
        params = {p.name for p in typer.main.get_command(app).commands["run"].params}

        assert "max_steps" in params

    def test_run_monta_a_estrategia_agentica_com_o_orcamento(self):
        with (
            patch("cli.main.AgenticPipelineStrategy") as strategy,
            patch("cli.main.ExperimentRunner") as experiment_runner,
        ):
            experiment_runner.return_value.execute.return_value = MagicMock(
                metrics={}, backlog_path="b.csv", failed_path=None, actions_path="a.jsonl"
            )
            result = CliRunner().invoke(
                app,
                [
                    "run",
                    "--strategy",
                    "pipeline_agentic",
                    "--n",
                    "1",
                    "--max-steps",
                    "7",
                    "--no-preflight",
                ],
            )

        assert result.exit_code == 0, result.output
        assert strategy.call_args.kwargs["max_steps"] == 7
        assert strategy.call_args.kwargs["protocol"] is CoordinationProtocol.FIXED
        assert "a.jsonl" in result.output

    @pytest.mark.parametrize(
        ("name", "protocol"),
        [
            ("pipeline_agentic", CoordinationProtocol.FIXED),
            ("pipeline_agentic_p2", CoordinationProtocol.PEERS),
            ("pipeline_agentic_p3", CoordinationProtocol.MEDIATED),
        ],
    )
    def test_cada_nome_de_estrategia_agentica_escolhe_o_seu_protocolo(self, name, protocol):
        with (
            patch("cli.main.AgenticPipelineStrategy") as strategy,
            patch("cli.main.ExperimentRunner") as experiment_runner,
        ):
            experiment_runner.return_value.execute.return_value = MagicMock(
                metrics={}, backlog_path="b.csv", failed_path=None, actions_path=None
            )
            result = CliRunner().invoke(
                app, ["run", "--strategy", name, "--n", "1", "--no-preflight"]
            )

        assert result.exit_code == 0, result.output
        assert strategy.call_args.kwargs["protocol"] is protocol
        assert experiment_runner.return_value.execute.call_args.args[1].strategy_name == name

    def test_estrategia_desconhecida_lista_os_protocolos_disponiveis(self):
        result = CliRunner().invoke(app, ["run", "--strategy", "pipeline_agentic_p1", "--n", "1"])

        assert result.exit_code != 0
        assert "pipeline_agentic_p2" in result.output and "pipeline_agentic_p3" in result.output
