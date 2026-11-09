"""Building blocks of the confirmatory protocol analysis (ADR-014)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef

from domain.agent_actions import ActionKind, ActionRecord, AgentEpisode, Goal
from domain.enums import MoSCoWPriority, RequirementType
from domain.models import PrioritizedRequirement, Requirement
from evaluation.agent_activity import cost_per_requirement, summarize_episodes
from evaluation.metrics import corpus
from evaluation.metrics.prioritization import compute_priority_reference_metrics
from evaluation.metrics.statistical import (
    friedman_test,
    holm_adjust,
    paired_permutation_test,
)
from experiments import confirmatory


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    reference = rng.integers(0, 4, size=200)
    predicted = np.where(rng.random(200) < 0.6, reference, rng.integers(0, 4, size=200))
    return reference, predicted


class TestCorpusMetrics:
    def test_f1_macro_igual_ao_do_scikit_learn(self, labels):
        reference, predicted = labels
        expected = f1_score(reference, predicted, average="macro")
        assert corpus.f1_macro(reference, predicted, 4) == pytest.approx(expected)

    def test_f1_macro_ignora_classe_que_nao_aparece_em_nenhum_lado(self):
        reference, predicted = np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])
        expected = f1_score(reference, predicted, average="macro")
        assert corpus.f1_macro(reference, predicted, 4) == pytest.approx(expected)

    @pytest.mark.parametrize("weights", [None, "linear"])
    def test_kappa_igual_ao_do_scikit_learn(self, labels, weights):
        reference, predicted = labels
        expected = cohen_kappa_score(reference, predicted, weights=weights)
        assert corpus.cohen_kappa(reference, predicted, 4, weights) == pytest.approx(expected)

    def test_mcc_igual_ao_do_scikit_learn(self, labels):
        reference, predicted = labels
        assert corpus.matthews(reference, predicted, 4) == pytest.approx(
            matthews_corrcoef(reference, predicted)
        )

    def test_kappa_indefinido_quando_so_ha_uma_classe(self):
        ones = np.ones(5, dtype=int)
        assert np.isnan(corpus.cohen_kappa(ones, ones, 4))


class TestPriorityGoldMetrics:
    def test_traz_f1_macro_e_kappa_nominal_e_marca_o_mapeamento_nao_verificado(self):
        M, S = MoSCoWPriority.MUST_HAVE, MoSCoWPriority.SHOULD_HAVE
        requirements = [
            Requirement(id=f"r{i}", text=f"Requisito {i}", metadata={"ground_truth_priority": gold})
            for i, gold in enumerate([M, M, S, S])
        ]
        predictions = [
            PrioritizedRequirement(
                id=f"r{i}",
                text=f"Requisito {i}",
                requirement_type=RequirementType.FUNCTIONAL,
                confidence=0.9,
                priority=priority,
                priority_score=priority.score,
                priority_rank=i + 1,
            )
            for i, priority in enumerate([M, S, S, S])
        ]
        result = compute_priority_reference_metrics(predictions, requirements)
        assert result["f1_macro"] is not None
        assert result["kappa_nominal"] is not None
        assert result["ordinal_mapping_verified"] is False


class TestPairedPermutationTest:
    def test_sem_diferenca_o_p_valor_e_grande(self):
        rng = np.random.default_rng(1)
        reference = rng.integers(0, 2, size=100)
        first = np.where(rng.random(100) < 0.7, reference, 1 - reference)
        second = first.copy()

        def difference(mask):
            a, b = np.where(mask, second, first), np.where(mask, first, second)
            return corpus.accuracy(reference, a, 2) - corpus.accuracy(reference, b, 2)

        result = paired_permutation_test(difference, 100, n_permutations=500)
        assert result.observed_difference == 0.0
        assert result.p_value == 1.0

    def test_diferenca_clara_da_p_valor_pequeno(self):
        reference = np.zeros(80, dtype=int)
        first, second = reference.copy(), np.ones(80, dtype=int)

        def difference(mask):
            a, b = np.where(mask, second, first), np.where(mask, first, second)
            return corpus.accuracy(reference, a, 2) - corpus.accuracy(reference, b, 2)

        result = paired_permutation_test(difference, 80, n_permutations=500)
        assert result.observed_difference == 1.0
        assert result.p_value < 0.01

    def test_p_valor_nunca_e_zero(self):
        result = paired_permutation_test(lambda mask: float(mask.sum() == 0), 10, 50)
        assert result.p_value > 0

    def test_recusa_menos_de_dois_pares(self):
        with pytest.raises(ValueError, match="at least 2"):
            paired_permutation_test(lambda mask: 0.0, 1)


class TestFriedmanAndHolm:
    def test_condicoes_diferentes_sao_detectadas(self):
        rng = np.random.default_rng(2)
        base = rng.normal(size=60)
        result = friedman_test([base, base + 1.0, base + 2.0])
        assert result.p_value < 0.001

    def test_condicoes_identicas_dao_p_um(self):
        assert friedman_test([[1.0, 2.0, 3.0]] * 3).p_value == 1.0

    def test_exige_tres_condicoes(self):
        with pytest.raises(ValueError, match="at least 3"):
            friedman_test([[1.0, 2.0], [1.0, 3.0]])

    def test_holm_ajusta_em_ordem_e_preserva_a_ordem_dada(self):
        assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])

    def test_holm_nao_passa_de_um(self):
        assert holm_adjust([0.6, 0.7]) == [1.0, 1.0]


def _episode(requirement_id, steps, latency=10.0, actions_available=("submit_x",)):
    actions = [
        ActionRecord(
            step=step,
            kind=ActionKind.SUBMIT,
            name="submit_x",
            input_tokens=100,
            output_tokens=10,
            latency_ms=latency,
        )
        for step in range(1, steps + 1)
    ]
    return AgentEpisode(
        agent="classifier",
        requirement_id=requirement_id,
        model="m",
        goal=Goal(description="g"),
        actions=actions,
        available_actions=list(actions_available),
    )


class TestCostAndReliability:
    def test_custo_soma_os_episodios_do_mesmo_requisito(self):
        cost = cost_per_requirement([_episode("a", 2), _episode("a", 1), _episode("b", 1)])
        assert cost["a"] == {"llm_calls": 3.0, "tokens": 330.0, "latency_ms": 30.0}
        assert cost["b"]["llm_calls"] == 1.0

    def test_resumo_traz_termino_falhas_por_100_chamadas_e_acoes_oferecidas(self):
        block = summarize_episodes([_episode("a", 2), _episode("b", 2)])["classifier"]
        assert block["termination"]["budget_exhausted"] == 2
        assert block["failures_per_100_calls"] == 50.0
        assert block["available_actions"] == ["submit_x"]


SPEC_MAX_STEPS = confirmatory.load_spec()["max_steps"]


class TestConfirmatorySpec:
    VALUES = {
        "dataset_name": "priorespa",
        "data_lang": "es",
        "lang": "es",
        "seed": 42,
        "temperature": 0.0,
        "model": confirmatory.load_spec()["model"],
        "information_regime": "zero_shot",
    }

    def test_execucao_dentro_do_desenho_e_confirmatoria(self):
        result = confirmatory.assess(
            "pipeline_agentic_p2", self.VALUES, None, 401, max_steps=SPEC_MAX_STEPS
        )
        assert result["is_confirmatory"] is True
        assert result["deviations"] == []

    def test_orcamento_de_passos_diferente_do_spec_e_desvio(self):
        result = confirmatory.assess("pipeline_agentic", self.VALUES, None, 401, max_steps=5)

        assert not result["is_confirmatory"]
        assert any(d.startswith("max_steps") for d in result["deviations"])

    def test_execucao_agentica_sem_orcamento_registrado_e_desvio(self):
        result = confirmatory.assess("pipeline_agentic", self.VALUES, None, 401)

        assert any(d.startswith("max_steps") for d in result["deviations"])

    def test_controle_pipeline_nao_depende_do_orcamento(self):
        result = confirmatory.assess("pipeline", self.VALUES, None, 401, max_steps=None)

        assert result["is_confirmatory"] is True

    def test_o_controle_pipeline_tambem_pertence_ao_desenho(self):
        assert confirmatory.assess("pipeline", self.VALUES, None, 401)["is_confirmatory"]

    def test_amostra_menor_que_o_censo_e_exploratoria(self):
        result = confirmatory.assess("pipeline_agentic", self.VALUES, 10, 10)
        assert not result["is_confirmatory"]
        assert any("n_samples" in d for d in result["deviations"])

    def test_desvio_de_modelo_ou_idioma_e_apontado(self):
        values = {**self.VALUES, "model": "ollama/llama3.1:8b", "lang": "pt"}
        deviations = confirmatory.assess("pipeline_agentic", values, None, 401)["deviations"]
        assert any(d.startswith("model") for d in deviations)
        assert any(d.startswith("lang") for d in deviations)

    def test_estrategia_fora_do_desenho_e_apontada(self):
        result = confirmatory.assess("baseline", self.VALUES, None, 401)
        assert any(d.startswith("strategy") for d in result["deviations"])

    def test_o_hash_do_arquivo_e_estavel(self):
        assert confirmatory.spec_sha256() == confirmatory.spec_sha256()
        assert len(confirmatory.spec_sha256()) == 64


class TestAnalyzerGuards:
    @staticmethod
    def _run(manifest):
        from pathlib import Path

        from experiments.analyze_agency import AgencyRun

        return AgencyRun(Path("run"), manifest, {}, {}, [])

    def test_recusa_execucao_sem_bloco_confirmatorio(self):
        from experiments.analyze_protocols import NotConfirmatoryError, ensure_confirmatory

        with pytest.raises(NotConfirmatoryError, match="no confirmatory block"):
            ensure_confirmatory({"P0": self._run({})}, exploratory=False)

    def test_recusa_execucao_sob_outra_versao_do_desenho(self):
        from experiments.analyze_protocols import NotConfirmatoryError, ensure_confirmatory

        manifest = {"confirmatory": {"is_confirmatory": True, "spec_sha256": "outro"}}
        with pytest.raises(NotConfirmatoryError, match="different confirmatory spec"):
            ensure_confirmatory({"P0": self._run(manifest)}, exploratory=False)

    def test_modo_exploratorio_dispensa_a_checagem(self):
        from experiments.analyze_protocols import ensure_confirmatory

        ensure_confirmatory({"P0": self._run({})}, exploratory=True)

    def test_recusa_misturar_execucoes_com_modelos_diferentes(self):
        from experiments.analyze_protocols import UnpairedProtocolsError, ensure_same_design

        base = {"dataset_name": "priorespa", "data_lang": "es", "lang": "es", "model": "a"}
        runs = {"P0": self._run(base), "P2": self._run({**base, "model": "b"})}
        with pytest.raises(UnpairedProtocolsError, match="model"):
            ensure_same_design(runs)


class TestNoiseFloor:
    def test_marca_o_contraste_que_supera_o_ruido(self):
        from experiments.analyze_protocols import annotate_with_noise

        row_big, row_small = {"delta": 0.10}, {"delta": 0.01}
        quality = {"tasks": {"type": {"f1_macro": {"P2-P0": row_big, "P3-P0": row_small}}}}
        noise = {"quality": {"tasks": {"type": {"f1_macro": {"P0R-P0": {"delta": -0.02}}}}}}
        annotate_with_noise(quality, noise)
        assert row_big["exceeds_noise_floor"] is True
        assert row_small["exceeds_noise_floor"] is False
        assert row_big["noise_floor_delta"] == -0.02

    def test_confirmatoria_sem_a_repeticao_do_p0_e_recusada(self):
        from experiments.analyze_protocols import NotConfirmatoryError, analyze

        with pytest.raises(NotConfirmatoryError, match="repeated P0"):
            analyze({}, p0_repeat=None, exploratory=False)


class TestMessageLog:
    @staticmethod
    def _messages():
        from domain.messages import Message, Performative, ReprioritizationRequest, RequestReply

        request = Message(
            sender="mediator",
            receiver="prioritizer",
            performative=Performative.REQUEST,
            requirement_id="r1",
            content=ReprioritizationRequest(reason="why", evidence="40 de 50 eram Must"),
        )
        reply = Message(
            sender="prioritizer",
            receiver="mediator",
            performative=Performative.REFUSE,
            requirement_id="r1",
            content=RequestReply(reason="keeps it"),
            in_reply_to=request.id,
        )
        return [request, reply]

    def test_sem_mensagens_nao_cria_arquivo(self, tmp_path):
        from evaluation.agent_activity import write_messages_jsonl

        assert write_messages_jsonl([], tmp_path / "messages.jsonl") is None
        assert not (tmp_path / "messages.jsonl").exists()

    def test_ida_e_volta_preserva_evidencia_e_resposta(self, tmp_path):
        from domain.messages import Message
        from evaluation.agent_activity import write_messages_jsonl

        path = write_messages_jsonl(self._messages(), tmp_path / "messages.jsonl")
        loaded = [
            Message.model_validate_json(line) for line in path.read_text("utf-8").splitlines()
        ]
        assert loaded[0].content.evidence == "40 de 50 eram Must"
        assert loaded[1].in_reply_to == loaded[0].id

    def test_fluxos_contam_remetente_destinatario_ato_e_conteudo(self):
        from evaluation.agent_activity import message_flows

        assert message_flows(self._messages()) == {
            "mediator>prioritizer:request:reprioritization_request": 1,
            "prioritizer>mediator:refuse:request_reply": 1,
        }


class TestRunTag:
    @staticmethod
    def _run(tmp_path, tag):
        from unittest.mock import MagicMock, patch

        from domain.models import PipelineState, Requirement
        from experiments.runner import ExperimentRunner, RunConfig
        from experiments.strategy import OrchestrationStrategy

        class Plain(OrchestrationStrategy):
            @property
            def name(self) -> str:
                return "plain"

            def execute(self, requirements, trace_writer=None) -> PipelineState:
                return PipelineState(raw_requirements=requirements)

        adapter = MagicMock()
        adapter.load.return_value = [Requirement(id="r1", text="Requisito de teste um")]
        with patch("experiments.runner.build_adapter", return_value=adapter):
            runner = ExperimentRunner(out_dir=str(tmp_path), trace_dir=str(tmp_path / "t"))
            config = RunConfig(
                strategy_name="plain",
                model="m",
                dataset_path="datasets/data/promise_nfr/promise_nfr_pt.csv",
                lang="pt",
                dataset_name="priorespa",
                data_lang="es",
                tag=tag,
            )
            return runner.execute(Plain(), config)

    def test_pasta_traz_idioma_dos_dados_do_prompt_e_o_rotulo(self, tmp_path):
        result = self._run(tmp_path, "p0-repeat")

        assert "_es-pt_n1_p0-repeat_" in result.run_id
        assert result.manifest["tag"] == "p0-repeat"

    def test_sem_rotulo_a_pasta_nao_ganha_segmento_extra(self, tmp_path):
        result = self._run(tmp_path, None)

        assert "_es-pt_n1_" in result.run_id and "None" not in result.run_id
        assert result.manifest["tag"] is None

    def test_rotulo_invalido_e_recusado(self):
        from experiments.runner import check_tag

        for bad in ("P0 Repeat", "-x", "a/b", ""):
            with pytest.raises(ValueError, match="tag must be"):
                check_tag(bad)

    def test_cli_recusa_rotulo_invalido(self):
        from typer.testing import CliRunner

        from cli.main import app

        result = CliRunner().invoke(
            app, ["run", "--strategy", "pipeline", "--tag", "Meu Rotulo", "--n", "1"]
        )
        assert result.exit_code == 2
        assert "tag must be" in result.output

    def test_assess_aceita_o_rotulo_do_piso_de_ruido_so_no_p0(self):
        values = TestConfirmatorySpec.VALUES
        ok = confirmatory.assess(
            "pipeline_agentic", values, None, 401, tag="p0-repeat", max_steps=SPEC_MAX_STEPS
        )
        wrong_arm = confirmatory.assess("pipeline_agentic_p2", values, None, 401, tag="p0-repeat")
        other = confirmatory.assess("pipeline_agentic", values, None, 401, tag="outro")
        assert ok["is_confirmatory"] is True
        assert any(d.startswith("tag") for d in wrong_arm["deviations"])
        assert any(d.startswith("tag") for d in other["deviations"])


class TestFindArms:
    @staticmethod
    def _write(root, name, strategy, tag=None, confirmatory_ok=True, sha=None):
        import json

        folder = root / name
        folder.mkdir()
        manifest = {
            "strategy": strategy,
            "tag": tag,
            "confirmatory": {
                "is_confirmatory": confirmatory_ok,
                "spec_sha256": sha or confirmatory.spec_sha256(),
            },
        }
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _all_arms(self, root):
        self._write(root, "a-p0", "pipeline_agentic")
        self._write(root, "a-p0r", "pipeline_agentic", tag="p0-repeat")
        self._write(root, "a-p2", "pipeline_agentic_p2")
        self._write(root, "a-p3", "pipeline_agentic_p3")

    def test_acha_os_quatro_bracos_e_distingue_a_repeticao(self, tmp_path):
        from experiments.analyze_protocols import find_arms

        self._all_arms(tmp_path)
        self._write(tmp_path, "controle", "pipeline")

        arms = find_arms(tmp_path)

        assert {k: v.name for k, v in arms.items()} == {
            "P0": "a-p0",
            "P0R": "a-p0r",
            "P2": "a-p2",
            "P3": "a-p3",
        }

    def test_ignora_exploratorias_e_outra_versao_do_desenho(self, tmp_path):
        from experiments.analyze_protocols import find_arms

        self._all_arms(tmp_path)
        self._write(tmp_path, "smoke", "pipeline_agentic", confirmatory_ok=False)
        self._write(tmp_path, "velho", "pipeline_agentic_p2", sha="outro")

        assert find_arms(tmp_path)["P2"].name == "a-p2"

    def test_recusa_quando_ha_dois_candidatos_ao_mesmo_braco(self, tmp_path):
        from experiments.analyze_protocols import ArmsNotFoundError, find_arms

        self._all_arms(tmp_path)
        self._write(tmp_path, "outro-p3", "pipeline_agentic_p3")

        with pytest.raises(ArmsNotFoundError, match="P3: a-p3, outro-p3"):
            find_arms(tmp_path)

    def test_recusa_quando_falta_um_braco(self, tmp_path):
        from experiments.analyze_protocols import ArmsNotFoundError, find_arms

        self._write(tmp_path, "a-p0", "pipeline_agentic")

        with pytest.raises(ArmsNotFoundError, match="P0R: no run"):
            find_arms(tmp_path)

    def test_caminho_explicito_nao_pode_trocar_a_repeticao_com_o_p0(self):
        from experiments.analyze_protocols import NotConfirmatoryError, check_arm_tags

        untagged = TestAnalyzerGuards._run({"tag": None})
        tagged = TestAnalyzerGuards._run({"tag": "p0-repeat"})

        check_arm_tags({"P0": untagged}, tagged)
        with pytest.raises(NotConfirmatoryError, match="must have been run"):
            check_arm_tags({"P0": untagged}, untagged)
        with pytest.raises(NotConfirmatoryError, match="carries a tag"):
            check_arm_tags({"P0": tagged}, tagged)


class TestDecisionsAndInteractionEffect:
    @staticmethod
    def _items():
        from domain.enums import MoSCoWPriority, RequirementType
        from domain.models import PrioritizedRequirement

        def item(rid, priority, reference):
            return PrioritizedRequirement(
                id=rid,
                text="Requisito de teste",
                requirement_type=RequirementType.FUNCTIONAL,
                confidence=0.9,
                priority=priority,
                priority_score=priority.score,
                priority_rank=1,
                metadata={"label_type": "F", "ground_truth_priority": reference},
            )

        must, should = MoSCoWPriority.MUST_HAVE, MoSCoWPriority.SHOULD_HAVE
        return [item("a", must, should), item("b", should, should)]

    def test_snapshot_e_linhas_trazem_inicial_final_e_referencia(self):
        from evaluation.decisions import build_decisions, snapshot_decisions

        items = self._items()
        initial = snapshot_decisions(items)
        revised = [items[0].model_copy(update={"priority": items[1].priority}), items[1]]

        rows = build_decisions(initial, revised)

        assert rows[0] == {
            "id": "a",
            "initial_type": "F",
            "final_type": "F",
            "initial_priority": "M",
            "final_priority": "S",
            "reference_type": "F",
            "reference_priority": "S",
        }

    def test_sem_primeiro_passe_nao_cria_arquivo(self, tmp_path):
        from evaluation.decisions import write_decisions_jsonl

        assert write_decisions_jsonl([], tmp_path / "decisions.jsonl") is None
        assert not (tmp_path / "decisions.jsonl").exists()

    def test_melhorou_piorou_e_neutro_so_contam_mudancas(self):
        from experiments.analyze_protocols import _agreement_changes

        rows = [
            {"i": "M", "f": "S", "r": "S"},
            {"i": "S", "f": "M", "r": "S"},
            {"i": "M", "f": "C", "r": "S"},
            {"i": "S", "f": "S", "r": "S"},
            {"i": "M", "f": "S", "r": None},
        ]

        result = _agreement_changes(rows, "i", "f", "r")

        assert result["measured"] == 4 and result["changed"] == 3
        assert (result["improved"], result["worsened"], result["neutral"]) == (1, 1, 1)
        assert result["p_improved_vs_worsened"] == 1.0

    def test_so_melhoras_dao_p_valor_pequeno(self):
        from experiments.analyze_protocols import _agreement_changes

        rows = [{"i": "M", "f": "S", "r": "S"}] * 12

        assert _agreement_changes(rows, "i", "f", "r")["p_improved_vs_worsened"] < 0.01

    def test_execucao_sem_primeiro_passe_e_avisada(self):
        from experiments.analyze_protocols import interaction_effect

        report = interaction_effect({"P0": TestAnalyzerGuards._run({})})

        assert "no first-pass" in report["P0"]["verdict"]

    def test_custo_por_fase_separa_a_coordenacao_do_trabalho(self):
        from domain.agent_actions import EpisodePhase
        from evaluation.agent_activity import cost_by_phase

        work = _episode("a", 1)
        reply = _episode("a", 1).model_copy(update={"phase": EpisodePhase.REPLY})

        phases = cost_by_phase([work, reply])

        assert phases["work"]["tokens"] == 110 and phases["reply"]["tokens"] == 110
        assert phases["coordination_token_share"]["value"] == 0.5

    def test_p_valor_compara_melhoras_e_pioras(self):
        from experiments.analyze_protocols import _agreement_changes

        better = [{"i": "M", "f": "S", "r": "S"}] * 5
        worse = [{"i": "S", "f": "M", "r": "S"}]

        result = _agreement_changes(better + worse, "i", "f", "r")

        assert result["p_improved_vs_worsened"] == pytest.approx(0.21875)
