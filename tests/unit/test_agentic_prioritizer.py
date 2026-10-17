from __future__ import annotations

from unittest.mock import patch

from agents.agentic.prioritizer import AgenticPrioritizer
from domain.agent_actions import AgentEpisode, EpisodeOutcome
from domain.enums import Lang, MoSCoWPriority, RequirementType
from domain.messages import (
    ClassificationReport,
    Message,
    Performative,
    ReclassificationRequest,
    inbox_of,
)
from domain.models import ClassifiedRequirement, PipelineState, Requirement
from tests.helpers.scripted_llm import ScriptedChatModel, text_reply, tool_call


def _prioritizer(script: list, **kwargs) -> tuple[AgenticPrioritizer, ScriptedChatModel]:
    llm = ScriptedChatModel(script=script)
    with patch("agents.agentic.prioritizer.build_llm", return_value=llm):
        return AgenticPrioritizer(model="ollama/fake", **kwargs), llm


def _submit(priority: str, score: float | None = None, why: str = "ok"):
    arguments = {"priority": priority, "justification": why}
    if score is not None:
        arguments["priority_score"] = score
    return tool_call("submit_priority", **arguments)


def _inform(
    req_id: str = "r1",
    text: str = "O sistema deve responder em 200ms",
    requirement_type: RequirementType = RequirementType.NON_FUNCTIONAL,
    category: str | None = "PE",
    confidence: float = 0.9,
    why: str = "tempo de resposta",
) -> Message:
    return Message(
        sender="classifier",
        receiver="prioritizer",
        performative=Performative.INFORM,
        requirement_id=req_id,
        content=ClassificationReport(
            text=text,
            requirement_type=requirement_type,
            nfr_category=category,
            confidence=confidence,
            justification=why,
        ),
    )


def _state(*messages: Message, **fields) -> PipelineState:
    return PipelineState(messages=list(messages), **fields)


class TestPrioritization:
    def test_prioriza_a_partir_das_mensagens_do_classificador(self):
        agent, _ = _prioritizer([_submit("M", 0.95, "crítico")])
        state = _state(_inform())

        agent.run(state)

        [item] = state.prioritized_requirements
        assert item.id == "r1" and item.priority is MoSCoWPriority.MUST_HAVE
        assert (item.priority_score, item.priority_justification) == (0.95, "crítico")
        assert item.requirement_type is RequirementType.NON_FUNCTIONAL
        assert (item.nfr_category, item.confidence) == ("PE", 0.9)

    def test_score_omitido_assume_o_da_prioridade(self):
        agent, _ = _prioritizer([_submit("S")])
        state = _state(_inform())

        agent.run(state)

        assert state.prioritized_requirements[0].priority_score == 0.75

    def test_ranking_global_pelo_score(self):
        agent, _ = _prioritizer([_submit("C"), _submit("M"), _submit("S")])
        state = _state(_inform("a"), _inform("b"), _inform("c"))

        agent.run(state)

        ranked = sorted(state.prioritized_requirements, key=lambda r: r.priority_rank or 0)
        assert [(r.id, r.priority_rank) for r in ranked] == [("b", 1), ("c", 2), ("a", 3)]

    def test_so_a_mensagem_mais_recente_de_cada_requisito_conta(self):
        agent, llm = _prioritizer([_submit("M")])
        state = _state(_inform("r1", confidence=0.4), _inform("r1", confidence=0.95))

        agent.run(state)

        assert len(state.prioritized_requirements) == 1
        assert state.prioritized_requirements[0].confidence == 0.95
        assert "confiança=0.95" in llm.seen[0][1].content

    def test_ignora_mensagens_de_outros_destinatarios(self):
        agent, _ = _prioritizer([])
        other = _inform().model_copy(update={"receiver": "outro"})

        agent.run(_state(other))

    def test_origem_e_metadados_vem_do_registro_do_ambiente(self):
        agent, _ = _prioritizer([_submit("M")])
        source = Requirement(
            id="r1",
            text="O sistema deve responder em 200ms",
            source="P1",
            metadata={"label_type": "NF"},
        )
        state = _state(_inform(), raw_requirements=[source])

        agent.run(state)

        item = state.prioritized_requirements[0]
        assert item.source == "P1" and item.metadata == {"label_type": "NF"}


class TestPerceptionIsOnlyTheMessage:
    """C3: what the prioritizer knows about the classifier's work is the message."""

    def test_ignora_a_saida_do_classificador_que_esta_no_estado(self):
        agent, llm = _prioritizer([_submit("M")])
        state = _state(_inform(category="PE", confidence=0.9))
        state.classified_requirements = [
            ClassifiedRequirement(
                id="r1",
                text="O sistema deve responder em 200ms",
                requirement_type=RequirementType.FUNCTIONAL,
                confidence=0.11,
            )
        ]

        agent.run(state)

        prompt = llm.seen[0][1].content
        assert "categoria=PE" in prompt and "0.11" not in prompt
        assert state.prioritized_requirements[0].requirement_type is RequirementType.NON_FUNCTIONAL

    def test_prompt_traz_confianca_e_justificativa_do_classificador(self):
        agent, llm = _prioritizer([_submit("M")])

        agent.run(_state(_inform(confidence=0.62, why="pode ser funcional")))

        prompt = llm.seen[0][1].content
        assert "confiança=0.62" in prompt and "pode ser funcional" in prompt

    def test_crenca_registra_de_qual_mensagem_veio(self):
        agent, _ = _prioritizer([_submit("M")])
        message = _inform()
        state = _state(message)

        agent.run(state)

        episode = AgentEpisode.model_validate(state.metrics["agent_episodes"][0])
        assert episode.beliefs[0].source == f"message:{message.id}"


class TestTools:
    def test_backlog_overview_descreve_o_lote(self):
        agent, _ = _prioritizer(
            [
                tool_call("backlog_overview"),
                _submit("M"),
                tool_call("backlog_overview"),
                _submit("S"),
            ]
        )
        state = _state(
            _inform("a", category="PE"),
            _inform("b", requirement_type=RequirementType.FUNCTIONAL, category=None),
        )

        agent.run(state)

        episodes = [AgentEpisode.model_validate(e) for e in state.metrics["agent_episodes"]]
        first = episodes[0].actions[0].observation
        assert "Batch of 2 requirements" in first
        assert "F: 1, NF: 1" in first and "PE: 1" in first and "none yet" in first
        assert "M: 1" in episodes[1].actions[0].observation

    def test_recall_lembra_prioridades_anteriores(self):
        agent, _ = _prioritizer(
            [
                _submit("M", why="tempo de resposta crítico"),
                tool_call("recall_similar_cases", query="responder rapidamente ao usuário"),
                _submit("M"),
            ]
        )
        state = _state(
            _inform("a", text="O sistema deve responder ao usuário rapidamente"),
            _inform("b", text="O sistema deve responder ao usuário em 200ms"),
        )

        agent.run(state)

        second = AgentEpisode.model_validate(state.metrics["agent_episodes"][1])
        assert "responder ao usuário rapidamente" in second.actions[0].observation

    def test_pedido_de_reclassificacao_vira_mensagem_para_o_classificador(self):
        agent, _ = _prioritizer(
            [
                tool_call("request_reclassification", reason="parece funcional"),
                _submit("C"),
            ]
        )
        state = _state(_inform("r1"))

        agent.run(state)

        [request] = inbox_of(state.messages, "classifier", Performative.REQUEST)
        assert request.sender == "prioritizer" and request.requirement_id == "r1"
        assert isinstance(request.content, ReclassificationRequest)
        assert request.content.reason == "parece funcional"
        assert state.prioritized_requirements[0].priority is MoSCoWPriority.COULD_HAVE

    def test_cada_pedido_leva_o_requisito_do_episodio_em_que_foi_feito(self):
        agent, _ = _prioritizer(
            [
                _submit("M"),
                tool_call("request_reclassification", reason="b duvidoso"),
                _submit("S"),
            ]
        )
        state = _state(_inform("a"), _inform("b"))

        agent.run(state)

        [request] = inbox_of(state.messages, "classifier", Performative.REQUEST)
        assert request.requirement_id == "b"

    def test_destinatario_do_pedido_e_configuravel(self):
        agent, _ = _prioritizer(
            [tool_call("request_reclassification", reason="x"), _submit("M")], peer="revisor"
        )
        state = _state(_inform())

        agent.run(state)

        assert inbox_of(state.messages, "revisor", Performative.REQUEST)


class TestBudgetAndEpisodes:
    def test_sem_resposta_no_orcamento_o_item_nao_e_priorizado(self):
        agent, _ = _prioritizer([text_reply("hmm"), text_reply("hmm")], max_steps=2)
        state = _state(_inform())

        agent.run(state)

        assert state.prioritized_requirements == []
        [record] = state.metrics["agent_episodes"]
        assert AgentEpisode.model_validate(record).outcome is EpisodeOutcome.BUDGET_EXHAUSTED
        assert state.metrics["failure_detections"][0]["severity"] == "fatal"

    def test_objetivo_no_prompt_e_no_episodio(self):
        agent, llm = _prioritizer([_submit("M")])
        state = _state(_inform())

        agent.run(state)

        episode = AgentEpisode.model_validate(state.metrics["agent_episodes"][0])
        assert episode.agent == "prioritizer"
        assert episode.goal.description in llm.seen[0][0].content

    def test_prompt_apresenta_as_ferramentas_sem_prescrever_procedimento(self):
        agent, llm = _prioritizer([_submit("M")], lang=Lang.EN)

        agent.run(_state(_inform()))

        system = llm.seen[0][0].content
        for name in (
            "recall_similar_cases",
            "backlog_overview",
            "request_reclassification",
            "submit_priority",
        ):
            assert name in system
        assert "first" not in system.lower().split() and "then" not in system.lower().split()
