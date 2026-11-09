"""Changing the model of the experiment touches one place: the provider registry, the preflight
and the frozen spec (ADR-014)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cli.main import app
from experiments import confirmatory
from llm import factory
from llm.preflight import preflight
from tests.helpers.scripted_llm import ScriptedChatModel, text_reply, tool_call


def _result(ok=True, tool_calling=True):
    return MagicMock(ok=ok, tool_calling=tool_calling, describe=lambda: "m: checked")


class TestProviderRegistry:
    def test_provedor_novo_funciona_sem_editar_o_build_llm(self):
        built = MagicMock()

        @factory.provider("fake")
        def _fake(name: str, temperature: float):
            built(name, temperature)
            return ScriptedChatModel(script=[])

        try:
            factory.build_llm("fake/meu-modelo", 0.2)
        finally:
            factory._PROVIDERS.pop("fake")

        built.assert_called_once_with("meu-modelo", 0.2)

    def test_prefixo_desconhecido_lista_os_validos(self):
        with pytest.raises(ValueError, match="foundry/.*ollama/"):
            factory.build_llm("inventado/modelo")

    def test_nome_sem_prefixo_conhecido_continua_indo_para_o_groq(self):
        with patch.dict(factory._PROVIDERS, {"groq": MagicMock(return_value="groq-model")}):
            assert factory.build_llm("llama-3.3-70b-versatile") == "groq-model"

    def test_prefixo_explicito_vence_o_nome_do_modelo(self):
        with patch.dict(factory._PROVIDERS, {"ollama": MagicMock(return_value="local")}):
            assert factory.build_llm("ollama/qwen2.5:7b") == "local"

    def test_provedores_de_fabrica_estao_registrados(self):
        registered = set(factory.available_providers())

        assert {"ollama", "azure", "foundry", "anthropic", "groq"} <= registered


class TestPreflight:
    def test_modelo_que_chama_ferramenta_passa(self):
        llm = ScriptedChatModel(script=[text_reply("OK"), tool_call("echo", text="hello")])

        result = preflight("fake/m", llm=llm)

        assert result.ok and result.tool_calling and result.error is None

    def test_modelo_que_responde_em_prosa_e_marcado_sem_tool_calling(self):
        llm = ScriptedChatModel(script=[text_reply("OK"), text_reply("vou chamar a ferramenta")])

        result = preflight("fake/m", llm=llm)

        assert result.ok and not result.tool_calling
        assert "NO tool calling" in result.describe()

    def test_falha_de_credencial_ou_cota_vira_resultado_com_a_causa(self):
        with patch("llm.preflight.build_llm", side_effect=RuntimeError("401 Unauthorized")):
            result = preflight("foundry/gpt-4.1-mini")

        assert not result.ok
        assert "RuntimeError: 401 Unauthorized" in result.describe()


class TestCheckModelCommand:
    def test_sai_com_erro_quando_o_modelo_nao_serve(self):
        with patch("cli.main.preflight", return_value=_result(ok=False, tool_calling=False)):
            result = CliRunner().invoke(app, ["check-model", "foundry/gpt-4.1-mini"])

        assert result.exit_code == 1

    def test_sai_com_sucesso_quando_serve(self):
        with patch("cli.main.preflight", return_value=_result()):
            result = CliRunner().invoke(app, ["check-model", "ollama/qwen2.5:7b"])

        assert result.exit_code == 0


class TestRunPreflight:
    @staticmethod
    def _invoke(args, preflight_result):
        with (
            patch("cli.main.preflight", return_value=preflight_result) as check,
            patch("cli.main.ExperimentRunner") as experiment_runner,
            patch("cli.main.AgenticPipelineStrategy"),
            patch("cli.main.BaselineStrategy"),
        ):
            experiment_runner.return_value.execute.return_value = MagicMock(
                metrics={}, backlog_path="b.csv", failed_path=None, actions_path=None
            )
            result = CliRunner().invoke(app, ["run", *args])
        return result, check, experiment_runner

    def test_modelo_inalcancavel_aborta_antes_de_rodar(self):
        result, _, runner = self._invoke(
            ["--strategy", "pipeline_agentic", "--n", "1"], _result(ok=False)
        )

        assert result.exit_code == 1
        runner.return_value.execute.assert_not_called()

    def test_agentico_exige_tool_calling(self):
        result, _, _ = self._invoke(
            ["--strategy", "pipeline_agentic", "--n", "1"], _result(tool_calling=False)
        )

        assert result.exit_code == 1

    def test_baseline_nao_exige_tool_calling(self):
        result, _, _ = self._invoke(
            ["--strategy", "baseline", "--n", "1"], _result(tool_calling=False)
        )

        assert result.exit_code == 0, result.output

    def test_modelo_igual_nos_dois_agentes_e_checado_uma_vez(self):
        _, check, _ = self._invoke(
            [
                "--strategy",
                "pipeline_agentic",
                "--n",
                "1",
                "--model",
                "ollama/m",
                "--pri-model",
                "ollama/m",
            ],
            _result(),
        )

        assert check.call_count == 1

    def test_no_preflight_nao_chama_o_modelo(self):
        _, check, _ = self._invoke(
            ["--strategy", "pipeline_agentic", "--n", "1", "--no-preflight"], _result()
        )

        check.assert_not_called()


class TestArmFromTheSpec:
    def test_valores_do_braco_vem_do_spec(self):
        spec = confirmatory.load_spec()

        values = confirmatory.arm_run_values("p2")

        assert values["strategy"] == "pipeline_agentic_p2"
        assert values["model"] == spec["model"].partition("+")[0]
        assert values["n"] is None and values["tag"] is None

    def test_repeticao_do_p0_e_o_p0_com_o_rotulo(self):
        values = confirmatory.arm_run_values("p0-repeat")

        assert values["strategy"] == "pipeline_agentic"
        assert values["tag"] == "p0-repeat"

    def test_controle_usa_o_pipeline(self):
        assert confirmatory.arm_run_values("control")["strategy"] == "pipeline"

    def test_braco_inexistente_e_recusado(self):
        with pytest.raises(ValueError, match="arm must be one of"):
            confirmatory.arm_run_values("p1")

    def test_trocar_o_modelo_do_spec_troca_o_braco_sem_outra_edicao(self):
        spec = {**confirmatory.load_spec(), "model": "foundry/x+foundry/y"}

        values = confirmatory.arm_run_values("p3", spec)

        assert (values["model"], values["pri_model"]) == ("foundry/x", "foundry/y")

    def test_cli_com_arm_executa_o_braco_exatamente_como_congelado(self):
        with (
            patch("cli.main.ExperimentRunner") as experiment_runner,
            patch("cli.main.AgenticPipelineStrategy") as patched_strategy,
        ):
            experiment_runner.return_value.execute.return_value = MagicMock(
                metrics={}, backlog_path="b.csv", failed_path=None, actions_path=None
            )
            result = CliRunner().invoke(app, ["run", "--arm", "p0-repeat", "--no-preflight"])

        config = experiment_runner.return_value.execute.call_args.args[1]
        assert result.exit_code == 0, result.output
        assert (config.tag, config.seed, config.lang) == ("p0-repeat", 42, "es")
        assert config.model == confirmatory.load_spec()["model"]
        assert config.n_samples is None
        assert config.max_steps == confirmatory.load_spec()["max_steps"]
        strategy_kwargs = patched_strategy.call_args.kwargs
        assert strategy_kwargs["max_steps"] == confirmatory.load_spec()["max_steps"]

    def test_cli_com_arm_recusa_flags_que_desviam_do_desenho(self):
        result = CliRunner().invoke(
            app, ["run", "--arm", "p0", "--model", "ollama/llama3.1:8b", "--no-preflight"]
        )

        assert result.exit_code == 2
        assert "--model" in result.output

    def test_cli_com_arm_recusa_trocar_o_orcamento_de_passos(self):
        result = CliRunner().invoke(
            app, ["run", "--arm", "p0", "--max-steps", "3", "--no-preflight"]
        )

        assert result.exit_code == 2
        assert "--max-steps" in result.output

    def test_cli_sem_strategy_nem_arm_pede_um_dos_dois(self):
        result = CliRunner().invoke(app, ["run", "--no-preflight"])

        assert result.exit_code == 2
        assert "--strategy" in result.output and "--arm" in result.output

    def test_manifest_traz_o_modelo_de_cada_agente(self):
        from experiments.runner import RunConfig

        config = RunConfig(strategy_name="s", model="foundry/a+ollama/b", dataset_path="x")

        assert config.models == {"classifier": "foundry/a", "prioritizer": "ollama/b"}
        single = RunConfig(strategy_name="s", model="ollama/a", dataset_path="x")
        assert single.models == {"classifier": "ollama/a", "prioritizer": "ollama/a"}


class TestDeepSeekProvider:
    def test_deepseek_esta_entre_os_provedores_registrados(self):
        assert "deepseek" in factory.available_providers()

    def test_constroi_o_cliente_com_endereco_e_chave_das_configuracoes(self):
        with (
            patch("langchain_openai.ChatOpenAI") as chat,
            patch.object(factory.settings, "deepseek_api_key", "chave-de-teste"),
            patch.object(factory.settings, "deepseek_base_url", "https://exemplo.test"),
        ):
            factory.build_llm("deepseek/deepseek-chat", 0.0)

        kwargs = chat.call_args.kwargs
        assert kwargs["model"] == "deepseek-chat"
        assert kwargs["base_url"] == "https://exemplo.test"
        assert kwargs["api_key"] == "chave-de-teste"
        assert kwargs["temperature"] == 0.0

    def test_configuracoes_trazem_o_endereco_padrao_e_chave_vazia(self):
        from config.settings import Settings

        fresh = Settings(_env_file=None, deepseek_api_key="")

        assert fresh.deepseek_base_url == "https://api.deepseek.com"
        assert fresh.deepseek_api_key == ""
