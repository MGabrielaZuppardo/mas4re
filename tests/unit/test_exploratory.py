"""Exploratory variants (ADR-016): apart from the confirmatory design, never mistaken for it."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from agents.memory import EpisodicMemory, MemoryItem, WarmupSeed
from cli.main import app
from domain.models import PipelineState, Requirement
from experiments import confirmatory
from experiments.exploratory import (
    manifest_block,
    variant_run_values,
    variant_spec,
    warmup_seed_from_run,
)
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import OrchestrationStrategy

# The confirmatory design is frozen: if this hash changes, an edit slipped into confirmatory.yml
# and every confirmatory run stops being recognised. Change it only on purpose (see ADR-014).
CONFIRMATORY_SPEC_SHA256 = "62db866413b128d58a2abce268870a7d4d2f680d50490f3345ed101a21824e2e"


class TestConfirmatoryDesignIsFrozen:
    def test_o_hash_do_spec_confirmatorio_nao_mudou(self):
        assert confirmatory.spec_sha256() == CONFIRMATORY_SPEC_SHA256


class TestVariantValues:
    def test_x1_herda_modelo_e_idiomas_do_desenho_confirmatorio(self):
        spec = confirmatory.load_spec()

        values = variant_run_values("x1-pinned-qwen", "p2")

        assert values["strategy"] == "pipeline_agentic_p2"
        assert values["model"] == spec["model"].partition("+")[0]
        assert (values["lang"], values["seed"], values["max_steps"]) == ("es", 42, 8)
        assert values["must_limit"] is None and values["pin_warmup_from"] == "P0R"
        assert values["tag"] == "x1-pinned-qwen"

    def test_x2_troca_so_o_limite_do_mediador(self):
        values = variant_run_values("x2-mediator-active", "p3")

        assert values["must_limit"] == 0.40
        assert values["strategy"] == "pipeline_agentic_p3"

    def test_x3_troca_o_modelo_e_a_repeticao_ganha_o_sufixo(self):
        values = variant_run_values("x3-gpt41mini", "p0-repeat")

        assert values["model"] == "foundry/gpt-4.1-mini"
        assert values["pin_warmup_from"] is None
        assert values["tag"] == "x3-gpt41mini-repeat"

    def test_braco_fora_da_variante_e_recusado(self):
        with pytest.raises(ValueError, match="arm must be one of"):
            variant_run_values("x2-mediator-active", "p2")

    def test_variante_inexistente_e_recusada(self):
        with pytest.raises(ValueError, match="variant must be one of"):
            variant_spec("x9")

    def test_bloco_do_manifest_traz_variante_hipotese_e_hash(self):
        block = manifest_block("x2-mediator-active", "p3", "run-ref")

        assert block["variant"] == "x2-mediator-active" and block["warmup_pinned_from"] == "run-ref"
        assert len(block["spec_sha256"]) == 64 and "0.40" in block["hypothesis"]


def _write_reference_run(root: Path) -> list[Requirement]:
    requirements = [Requirement(id=f"r{i}", text=f"Texto do requisito {i}") for i in range(4)]
    root.mkdir()
    episodes = [
        {
            "agent": "prioritizer",
            "phase": "work",
            "requirement_id": f"r{i}",
            "actions": [
                {"name": "recall_similar_cases", "arguments": {}},
                {"name": "submit_priority", "arguments": {"justification": f"porque {i}"}},
            ],
        }
        for i in range(4)
    ]
    (root / "actions.jsonl").write_text(
        "\n".join(json.dumps(e) for e in episodes), encoding="utf-8"
    )
    priorities = ["S", "M", "M", "C"]
    (root / "decisions.jsonl").write_text(
        "\n".join(
            json.dumps({"id": f"r{i}", "initial_priority": p}) for i, p in enumerate(priorities)
        ),
        encoding="utf-8",
    )
    return requirements


class TestWarmupSeed:
    def test_semente_traz_texto_prioridade_inicial_e_justificativa(self, tmp_path):
        requirements = _write_reference_run(tmp_path / "ref")

        seed = warmup_seed_from_run(tmp_path / "ref", requirements, size=3)

        assert [i.requirement_id for i in seed.items] == ["r0", "r1", "r2"]
        assert seed.items[1] == MemoryItem("r1", "Texto do requisito 1", "M", "porque 1")
        assert dict(seed.decided) == {"M": 2, "S": 1}

    def test_nao_passa_do_tamanho_do_aquecimento(self, tmp_path):
        requirements = _write_reference_run(tmp_path / "ref")

        assert len(warmup_seed_from_run(tmp_path / "ref", requirements, size=2).items) == 2

    def test_memoria_semeada_fica_congelada_e_responde_a_busca(self):
        memory = EpisodicMemory()
        seed = WarmupSeed(
            items=(MemoryItem("r0", "controle de acesso ao sistema", "M", "seguranca"),),
            decided=(("M", 1),),
        )

        memory.seed(seed)
        memory.add("r9", "controle de acesso novo", "S", "ignorado")

        assert memory.frozen
        assert [i.requirement_id for i in memory.retrieve_similar("acesso ao sistema")] == ["r0"]

    def test_priorizador_semeado_comeca_com_a_visao_do_aquecimento(self):
        from agents.agentic.prioritizer import AgenticPrioritizer
        from tests.helpers.scripted_llm import ScriptedChatModel

        seed = WarmupSeed(
            items=(MemoryItem("r0", "texto qualquer", "M", "j"),), decided=(("M", 1), ("S", 4))
        )
        with patch(
            "agents.agentic.prioritizer.build_llm", return_value=ScriptedChatModel(script=[])
        ):
            agent = AgenticPrioritizer(model="ollama/x", warmup_seed=seed)

        agent.prioritize_batch([])

        assert agent._memory is not None and agent._memory.frozen
        assert dict(agent._view.decided) == {"M": 1, "S": 4}


class _Plain(OrchestrationStrategy):
    @property
    def name(self) -> str:
        return "pipeline_agentic"

    def execute(self, requirements, trace_writer=None) -> PipelineState:
        return PipelineState(raw_requirements=requirements)


class TestRunIsLabelledExploratory:
    @staticmethod
    def _run(tmp_path, exploratory):
        adapter = MagicMock()
        adapter.load.return_value = [Requirement(id="r1", text="Requisito de teste um")]
        spec = confirmatory.load_spec()
        with patch("experiments.runner.build_adapter", return_value=adapter):
            runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "t"))
            return runner.execute(
                _Plain(),
                RunConfig(
                    strategy_name="pipeline_agentic",
                    model=spec["model"],
                    dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
                    lang="es",
                    dataset_name="priorespa",
                    data_lang="es",
                    max_steps=spec["max_steps"],
                    exploratory=exploratory,
                ),
            )

    def test_execucao_exploratoria_nunca_e_confirmatoria(self, tmp_path):
        result = self._run(tmp_path, manifest_block("x1-pinned-qwen", "p0", None))

        confirmatory_status = result.manifest["confirmatory"]
        assert confirmatory_status["is_confirmatory"] is False
        assert any("exploratory variant" in d for d in confirmatory_status["deviations"])
        assert result.manifest["exploratory"]["variant"] == "x1-pinned-qwen"

    def test_assess_com_variante_acusa_desvio(self):
        spec = confirmatory.load_spec()
        values = {
            "dataset_name": "priorespa",
            "data_lang": "es",
            "lang": "es",
            "seed": 42,
            "temperature": 0.0,
            "model": spec["model"],
            "information_regime": "zero_shot",
        }

        result = confirmatory.assess(
            "pipeline_agentic", values, None, 401, max_steps=8, exploratory_variant="x2"
        )

        assert result["is_confirmatory"] is False


class TestExploreCommand:
    def test_list_mostra_as_variantes_sem_rodar_nada(self):
        result = CliRunner().invoke(app, ["explore", "--list"])

        assert result.exit_code == 0
        assert "x1-pinned-qwen" in result.output and "x3-gpt41mini" in result.output

    def test_exige_variante_e_braco(self):
        result = CliRunner().invoke(app, ["explore", "--no-preflight"])

        assert result.exit_code == 2 and "--variant" in result.output

    def test_braco_que_a_variante_nao_tem_e_recusado(self):
        result = CliRunner().invoke(
            app, ["explore", "--variant", "x2-mediator-active", "--arm", "p2", "--no-preflight"]
        )

        assert result.exit_code == 2

    def test_grava_fora_de_experiments_results_e_com_o_limite_da_variante(self):
        with (
            patch("cli.main.ExperimentRunner") as experiment_runner,
            patch("cli.main.AgenticPipelineStrategy") as strategy,
            patch("cli.main.find_arms", return_value={"P0R": Path("ref")}),
            patch("cli.main.build_adapter"),
            patch("cli.main.warmup_seed_from_run", return_value="SEED"),
        ):
            experiment_runner.return_value.execute.return_value = MagicMock(run_id="x")
            result = CliRunner().invoke(
                app,
                ["explore", "--variant", "x2-mediator-active", "--arm", "p3", "--no-preflight"],
            )

        assert result.exit_code == 0, result.output
        out_dir = experiment_runner.call_args.kwargs["out_dir"]
        assert out_dir.replace("\\", "/").endswith("experiments/exploratory/results")
        assert strategy.call_args.kwargs["must_limit"] == 0.40
        assert strategy.call_args.kwargs["prioritizer_warmup_seed"] == "SEED"
        config = experiment_runner.return_value.execute.call_args.args[1]
        assert config.exploratory["variant"] == "x2-mediator-active"
        assert config.tag == "x2-mediator-active"


class TestExploratoryAnalysis:
    @staticmethod
    def _run(strategy_tag, priorities, model="m"):
        from experiments.analyze_agency import AgencyRun

        decisions = [{"id": f"r{i}", "initial_priority": p} for i, p in enumerate(priorities)]
        episodes = [_episode_of(f"r{i}") for i in range(len(priorities))]
        return AgencyRun(Path(strategy_tag), {"model": model}, {}, {}, episodes, [], decisions)

    def test_tabela_agrupa_execucoes_com_o_mesmo_aquecimento(self):
        from experiments.analyze_exploratory import warmup_table

        runs = {
            "P0": self._run("a", ["S", "S", "M"]),
            "P0R": self._run("b", ["S", "M", "M"]),
            "P3": self._run("c", ["S", "M", "M"]),
        }

        rows = {r["run"]: r for r in warmup_table(runs, size=3)}

        assert rows["P0R"]["same_warmup_as"] == ["P3"] and rows["P0"]["same_warmup_as"] == []
        assert rows["P0"]["warmup"] == "SSM"

    def test_aquecimento_respeita_o_tamanho(self):
        from experiments.analyze_exploratory import warmup_signature

        run = self._run("a", ["S", "M", "C", "W"])

        assert warmup_signature(run, 2) == ["S", "M"]

    def test_relatorio_descritivo_nao_traz_p_valores(self):
        from experiments.analyze_exploratory import _descriptive

        report = {
            "tasks": {
                "type": {"f1_macro": {"A-B": {"delta": 0.1, "p_permutation": 0.01, "p_holm": 0.03}}}
            }
        }

        row = _descriptive(report)["tasks"]["type"]["f1_macro"]["A-B"]

        assert row == {"delta": 0.1}

    def test_recusa_comparar_modelos_diferentes(self):
        from experiments.analyze_exploratory import _require_same_model

        runs = {
            "V:P0": self._run("a", ["S"], "foundry/x"),
            "R:P0": self._run("b", ["S"], "ollama/y"),
        }

        with pytest.raises(ValueError, match="different models"):
            _require_same_model(runs)

    def test_variante_sem_execucoes_e_recusada(self, tmp_path):
        from experiments.analyze_exploratory import load_variant_runs

        with pytest.raises(ValueError, match="no run of x2-mediator-active"):
            load_variant_runs("x2-mediator-active", tmp_path)

    def test_duas_execucoes_do_mesmo_braco_da_variante_sao_recusadas(self, tmp_path):
        from experiments.analyze_exploratory import load_variant_runs

        for name in ("a", "b"):
            folder = tmp_path / name
            folder.mkdir()
            (folder / "manifest.json").write_text(
                json.dumps({"exploratory": {"variant": "x1-pinned-qwen", "arm": "p2"}}),
                encoding="utf-8",
            )

        with pytest.raises(ValueError, match="more than one run"):
            load_variant_runs("x1-pinned-qwen", tmp_path)


def _episode_of(requirement_id):
    from domain.agent_actions import AgentEpisode, Goal

    return AgentEpisode(
        agent="prioritizer", requirement_id=requirement_id, model="m", goal=Goal(description="g")
    )


class TestLlamaVariant:
    def test_x4_usa_o_llama_nos_dois_agentes_e_so_p0_e_repeticao(self):
        values = variant_run_values("x4-llama31", "p0-repeat")

        assert (
            values["model"] == "ollama/llama3.1:8b" and values["pri_model"] == "ollama/llama3.1:8b"
        )
        assert values["tag"] == "x4-llama31-repeat" and values["pin_warmup_from"] is None
        assert variant_spec("x4-llama31")["arms"] == ["p0", "p0-repeat"]

    def test_x4_recusa_p2_enquanto_nao_for_liberado(self):
        with pytest.raises(ValueError, match="arm must be one of"):
            variant_run_values("x4-llama31", "p2")

    def test_hash_da_variante_nao_muda_quando_outra_variante_e_editada(self):
        from experiments.exploratory import load_exploratory_spec, variant_sha256

        spec = load_exploratory_spec()
        before = variant_sha256("x4-llama31", spec)
        spec["variants"]["x2-mediator-active"]["must_limit"] = 0.55

        assert variant_sha256("x4-llama31", spec) == before
        assert variant_sha256("x2-mediator-active", spec) != variant_sha256("x2-mediator-active")
