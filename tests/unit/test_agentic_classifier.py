from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.agentic.classifier import AgenticClassifier
from domain.agent_actions import ActionKind, AgentEpisode, EpisodeOutcome
from domain.enums import Lang, RequirementType
from domain.failures import FailureMode
from domain.messages import ClassificationReport, Performative, inbox_of
from domain.models import PipelineState, Requirement
from tests.helpers.scripted_llm import ScriptedChatModel, text_reply, tool_call


def _classifier(script: list, **kwargs) -> tuple[AgenticClassifier, ScriptedChatModel]:
    llm = ScriptedChatModel(script=script)
    with patch("agents.agentic.classifier.build_llm", return_value=llm):
        return AgenticClassifier(model="ollama/fake", **kwargs), llm


def _submit(requirement_type: str, category: str | None, confidence: float = 0.9, why: str = "ok"):
    return tool_call(
        "submit_classification",
        requirement_type=requirement_type,
        nfr_category=category,
        confidence=confidence,
        justification=why,
    )


def _requirement(
    text: str = "O sistema deve responder em 200ms", req_id: str = "r1"
) -> Requirement:
    return Requirement(id=req_id, text=text)


class TestClassification:
    def test_classifica_com_a_resposta_que_o_agente_submeteu(self):
        agent, _ = _classifier([_submit("NF", "PE", 0.92, "Desempenho.")])

        [item] = agent.classify_batch([_requirement()])

        assert item.requirement_type is RequirementType.NON_FUNCTIONAL
        assert (item.nfr_category, item.confidence, item.justification) == (
            "PE",
            0.92,
            "Desempenho.",
        )

    def test_categoria_em_prosa_e_normalizada_para_o_codigo(self):
        agent, _ = _classifier([_submit("NF", "desempenho")])

        [item] = agent.classify_batch([_requirement()])

        assert item.nfr_category == "PE"

    def test_o_agente_escolhe_se_e_quando_usar_ferramentas(self):
        agent, _ = _classifier(
            [
                _submit("F", None),
                tool_call("lookup_nfr_taxonomy", category_code="PE"),
                _submit("NF", "PE"),
            ]
        )

        agent.classify_batch([_requirement(req_id="a"), _requirement(req_id="b")])

        sequences = [e.action_sequence for e in agent._episodes]
        assert sequences == [
            ("submit_classification",),
            ("lookup_nfr_taxonomy", "submit_classification"),
        ]

    def test_a_ferramenta_de_taxonomia_devolve_a_definicao_do_dataset(self):
        agent, _ = _classifier(
            [tool_call("lookup_nfr_taxonomy", category_code="SE"), _submit("NF", "SE")],
            nfr_categories=[("SE", "Security — protege dados"), ("PE", "Performance")],
        )

        agent.classify_batch([_requirement()])

        assert "protege dados" in agent._episodes[0].actions[0].observation

    def test_verificador_externo_aponta_a_inconsistencia_e_o_agente_corrige(self):
        agent, _ = _classifier(
            [
                tool_call("check_consistency", requirement_type="NF"),
                _submit("NF", "PE"),
            ]
        )

        [item] = agent.classify_batch([_requirement()])

        observation = agent._episodes[0].actions[0].observation
        assert observation.startswith("Inconsistent") and "needs an nfr_category" in observation
        assert item.nfr_category == "PE"

    def test_resposta_em_texto_json_tambem_encerra_o_trabalho(self):
        agent, _ = _classifier(
            [
                text_reply(
                    '{"requirement_type": "F", "nfr_category": null, '
                    '"confidence": 0.8, "justification": "função"}'
                )
            ]
        )

        [item] = agent.classify_batch([_requirement()])

        assert item.requirement_type is RequirementType.FUNCTIONAL
        assert agent._episodes[0].submitted_via == "text"


class TestBudget:
    def test_sem_resposta_no_orcamento_o_item_falha_sem_rotulo_inventado(self):
        agent, llm = _classifier(
            [tool_call("lookup_nfr_taxonomy", category_code="PE")] * 3, max_steps=3
        )

        classified = agent.classify_batch([_requirement()])

        assert classified == []
        assert llm.calls == 3
        episode = agent._episodes[0]
        assert episode.outcome is EpisodeOutcome.BUDGET_EXHAUSTED and episode.submitted_via is None

    def test_a_falha_e_registrada_como_esquema_invalido_fatal(self):
        agent, _ = _classifier(
            [tool_call("lookup_nfr_taxonomy", category_code="PE")] * 2, max_steps=2
        )

        agent.classify_batch([_requirement()])

        assert [r.mode for r in agent._failure_records] == [FailureMode.SCHEMA_INVALID]


class TestRun:
    def test_run_publica_um_inform_por_requisito_para_o_par(self):
        agent, _ = _classifier([_submit("NF", "PE", 0.88, "Desempenho.")])
        state = PipelineState(raw_requirements=[_requirement()])

        agent.run(state)

        [message] = inbox_of(state.messages, "prioritizer", Performative.INFORM)
        assert (message.sender, message.requirement_id) == ("classifier", "r1")
        assert isinstance(message.content, ClassificationReport)
        assert message.content.confidence == 0.88
        assert message.content.nfr_category == "PE"
        assert message.content.text == "O sistema deve responder em 200ms"

    def test_o_destinatario_e_configuravel(self):
        agent, _ = _classifier([_submit("F", None)], peer="revisor")
        state = PipelineState(raw_requirements=[_requirement()])

        agent.run(state)

        assert state.messages[0].receiver == "revisor"

    def test_run_guarda_os_episodios_no_estado_e_esvazia_o_agente(self):
        agent, _ = _classifier([_submit("F", None)])
        state = PipelineState(raw_requirements=[_requirement()])

        agent.run(state)

        [record] = state.metrics["agent_episodes"]
        episode = AgentEpisode.model_validate(record)
        assert (episode.agent, episode.requirement_id) == ("classifier", "r1")
        assert episode.actions[0].kind is ActionKind.SUBMIT
        assert agent._episodes == []

    def test_requisito_perdido_fica_de_fora_dos_classificados_e_das_mensagens(self):
        agent, _ = _classifier(
            [_submit("F", None), text_reply("não sei"), text_reply("não sei")], max_steps=2
        )
        state = PipelineState(
            raw_requirements=[_requirement(req_id="ok"), _requirement("Outro requisito", "lost")]
        )

        agent.run(state)

        assert [c.id for c in state.classified_requirements] == ["ok"]
        assert [m.requirement_id for m in state.messages] == ["ok"]
        assert len(state.metrics["agent_episodes"]) == 2
        fatal = [d for d in state.metrics["failure_detections"] if d["severity"] == "fatal"]
        assert [d["requirement_id"] for d in fatal] == ["lost"]


class TestMemory:
    def test_o_agente_pode_consultar_o_que_ja_decidiu_com_confianca(self):
        agent, _ = _classifier(
            [
                _submit("NF", "PE", 0.95, "tempo de resposta"),
                tool_call("recall_similar_cases", query="O sistema deve responder rapidamente"),
                _submit("NF", "PE"),
            ]
        )

        agent.classify_batch(
            [
                _requirement("O sistema deve responder em 200ms", "a"),
                _requirement("O sistema deve responder rapidamente", "b"),
            ]
        )

        observation = agent._episodes[1].actions[0].observation
        assert "O sistema deve responder em 200ms" in observation and "NF/PE" in observation

    @pytest.mark.parametrize(
        ("category", "confidence"),
        [("PE", 0.5), (None, 0.95)],
        ids=["pouco confiante", "inconsistente"],
    )
    def test_resposta_duvidosa_nao_vira_exemplo(self, category, confidence):
        agent, _ = _classifier(
            [
                _submit("NF", category, confidence),
                tool_call("recall_similar_cases", query="O sistema deve responder rapidamente"),
                _submit("F", None),
            ]
        )

        agent.classify_batch(
            [
                _requirement("O sistema deve responder em 200ms", "a"),
                _requirement("O sistema deve responder rapidamente", "b"),
            ]
        )

        assert agent._episodes[1].actions[0].observation == "No similar cases handled so far."


class TestGoalBeliefsAndPrompt:
    def test_objetivo_esta_no_prompt_e_no_episodio(self):
        agent, llm = _classifier([_submit("F", None)])

        agent.classify_batch([_requirement()])

        system = llm.seen[0][0].content
        goal = agent._episodes[0].goal.description
        assert goal in system and "funcional" in goal

    def test_crenca_de_percepcao_esta_no_prompt_e_no_episodio(self):
        agent, llm = _classifier([_submit("F", None)])

        agent.classify_batch([_requirement("Texto do requisito de teste")])

        assert "Texto do requisito de teste" in llm.seen[0][1].content
        beliefs = agent._episodes[0].beliefs
        assert [(b.source, b.content) for b in beliefs] == [
            ("percept", "Texto do requisito de teste")
        ]

    def test_prompt_apresenta_todas_as_ferramentas_sem_prescrever_procedimento(self):
        agent, llm = _classifier([_submit("F", None)])

        agent.classify_batch([_requirement()])

        system = llm.seen[0][0].content
        for name in (
            "lookup_nfr_taxonomy",
            "recall_similar_cases",
            "check_consistency",
            "submit_classification",
        ):
            assert name in system
        assert "primeiro" not in system.lower() and "depois" not in system.lower()

    def test_prompt_em_ingles(self):
        agent, llm = _classifier([_submit("F", None)], lang=Lang.EN)

        agent.classify_batch([_requirement()])

        assert "Goal: Classify the requirement" in llm.seen[0][0].content
