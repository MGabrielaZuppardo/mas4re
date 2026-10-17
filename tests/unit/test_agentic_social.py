"""Social ability: an agent asks its peer for something, and the peer decides how to answer."""

from __future__ import annotations

from unittest.mock import patch

from agents.agentic.classifier import AgenticClassifier
from agents.agentic.prioritizer import AgenticPrioritizer
from domain.agent_actions import AgentEpisode, EpisodeOutcome
from domain.enums import RequirementType
from domain.messages import (
    ClassificationReport,
    Message,
    Performative,
    ReclassificationRequest,
    RequestReply,
    inbox_of,
)
from domain.models import ClassifiedRequirement, PipelineState, Requirement
from tests.helpers.scripted_llm import ScriptedChatModel, text_reply, tool_call

_TEXT = "O sistema deve permitir que o usuário exporte relatórios"


def _classifier(script: list, **kwargs) -> tuple[AgenticClassifier, ScriptedChatModel]:
    llm = ScriptedChatModel(script=script)
    with patch("agents.agentic.classifier.build_llm", return_value=llm):
        return AgenticClassifier(model="ollama/fake", **kwargs), llm


def _prioritizer(script: list) -> tuple[AgenticPrioritizer, ScriptedChatModel]:
    llm = ScriptedChatModel(script=script)
    with patch("agents.agentic.prioritizer.build_llm", return_value=llm):
        return AgenticPrioritizer(model="ollama/fake"), llm


def _request(reason: str = "parece funcional", req_id: str = "r1") -> Message:
    return Message(
        sender="prioritizer",
        receiver="classifier",
        performative=Performative.REQUEST,
        requirement_id=req_id,
        content=ReclassificationRequest(reason=reason),
    )


def _state_with_request(request: Message | None = None) -> PipelineState:
    classified = ClassifiedRequirement(
        id="r1",
        text=_TEXT,
        requirement_type=RequirementType.NON_FUNCTIONAL,
        nfr_category="US",
        confidence=0.6,
        justification="parece usabilidade",
    )
    return PipelineState(
        raw_requirements=[Requirement(id="r1", text=_TEXT)],
        classified_requirements=[classified],
        messages=[request or _request()],
    )


def _agree(**revised):
    arguments = {
        "decision": "agree",
        "reason": "tem razão",
        "requirement_type": "F",
        "nfr_category": None,
        "confidence": 0.85,
        "justification": "É uma função do sistema.",
        **revised,
    }
    return tool_call("reply_to_request", **arguments)


def _refuse(reason: str = "a classificação está correta"):
    return tool_call("reply_to_request", decision="refuse", reason=reason)


class TestAcceptingARequest:
    def test_aceita_revisa_a_classificacao_e_responde_com_o_relatorio_novo(self):
        agent, _ = _classifier([_agree()])
        request = _request()
        state = _state_with_request(request)

        agent.answer_requests(state)

        [reply] = inbox_of(state.messages, "prioritizer", Performative.AGREE)
        assert reply.in_reply_to == request.id and reply.sender == "classifier"
        assert isinstance(reply.content, RequestReply)
        assert reply.content.revised is not None
        assert reply.content.revised.requirement_type is RequirementType.FUNCTIONAL
        assert reply.content.revised.confidence == 0.85
        [current] = state.classified_requirements
        assert current.requirement_type is RequirementType.FUNCTIONAL
        assert current.nfr_category is None and current.justification == "É uma função do sistema."

    def test_aceitar_sem_a_classificacao_revista_e_acao_invalida_e_o_agente_corrige(self):
        agent, _ = _classifier(
            [tool_call("reply_to_request", decision="agree", reason="tem razão"), _agree()]
        )
        state = _state_with_request()

        agent.answer_requests(state)

        episode = AgentEpisode.model_validate(state.metrics["agent_episodes"][0])
        assert episode.actions[0].kind.value == "invalid"
        assert "requirement_type" in episode.actions[0].observation
        assert inbox_of(state.messages, "prioritizer", Performative.AGREE)


class TestRefusingARequest:
    def test_recusa_com_o_motivo_e_mantem_a_classificacao(self):
        agent, _ = _classifier([_refuse("o texto trata de exportação, é função")])
        state = _state_with_request()

        agent.answer_requests(state)

        [reply] = inbox_of(state.messages, "prioritizer", Performative.REFUSE)
        assert isinstance(reply.content, RequestReply) and reply.content.revised is None
        assert reply.content.reason == "o texto trata de exportação, é função"
        assert state.classified_requirements[0].requirement_type is RequirementType.NON_FUNCTIONAL

    def test_a_decisao_depende_do_que_o_agente_conclui_nao_do_pedido(self):
        accepting, _ = _classifier([_agree()])
        refusing, _ = _classifier([_refuse()])
        first, second = _state_with_request(), _state_with_request()

        accepting.answer_requests(first)
        refusing.answer_requests(second)

        assert first.messages[-1].performative is Performative.AGREE
        assert second.messages[-1].performative is Performative.REFUSE


class TestTheRequestIsAnEpisodeOfItsOwn:
    def test_percepcao_traz_o_pedido_e_a_classificacao_atual(self):
        agent, llm = _classifier([_refuse()])

        agent.answer_requests(_state_with_request(_request("parece funcional")))

        percept = llm.seen[0][1].content
        assert "parece funcional" in percept and "categoria=US" in percept

    def test_episodio_registra_objetivo_crencas_e_a_acao_terminal(self):
        agent, llm = _classifier([_refuse()])
        request = _request()
        state = _state_with_request(request)

        agent.answer_requests(state)

        episode = AgentEpisode.model_validate(state.metrics["agent_episodes"][0])
        assert episode.goal.description in llm.seen[0][0].content
        assert episode.action_sequence == ("reply_to_request",)
        assert f"message:{request.id}" in [b.source for b in episode.beliefs]

    def test_o_agente_pode_usar_ferramentas_antes_de_responder(self):
        agent, _ = _classifier([tool_call("check_consistency", requirement_type="F"), _agree()])
        state = _state_with_request()

        agent.answer_requests(state)

        episode = AgentEpisode.model_validate(state.metrics["agent_episodes"][0])
        assert episode.action_sequence == ("check_consistency", "reply_to_request")

    def test_pedido_ja_respondido_nao_e_respondido_de_novo(self):
        agent, llm = _classifier([_refuse()])
        state = _state_with_request()

        agent.answer_requests(state)
        agent.answer_requests(state)

        assert llm.calls == 1
        assert len(inbox_of(state.messages, "prioritizer", Performative.REFUSE)) == 1


class TestFailures:
    def test_requisito_desconhecido_recebe_failure_sem_gastar_o_modelo(self):
        agent, llm = _classifier([])
        state = _state_with_request(_request(req_id="inexistente"))

        agent.answer_requests(state)

        assert llm.calls == 0
        assert state.messages[-1].performative is Performative.FAILURE

    def test_sem_resposta_no_orcamento_recebe_failure(self):
        agent, _ = _classifier([text_reply("hmm"), text_reply("hmm")], max_steps=2)
        state = _state_with_request()

        agent.answer_requests(state)

        assert state.messages[-1].performative is Performative.FAILURE
        episode = AgentEpisode.model_validate(state.metrics["agent_episodes"][0])
        assert episode.outcome is EpisodeOutcome.BUDGET_EXHAUSTED


class TestTwoAgentsTalking:
    def test_priorizador_questiona_e_o_classificador_decide(self):
        classifier, _ = _classifier(
            [
                tool_call(
                    "submit_classification",
                    requirement_type="NF",
                    nfr_category="US",
                    confidence=0.6,
                    justification="parece usabilidade",
                ),
                _agree(),
            ]
        )
        prioritizer, _ = _prioritizer(
            [
                tool_call("request_reclassification", reason="parece funcional"),
                tool_call("submit_priority", priority="S", justification="provisório"),
            ]
        )
        state = PipelineState(raw_requirements=[Requirement(id="r1", text=_TEXT)])

        classifier.run(state)
        prioritizer.run(state)
        classifier.answer_requests(state)

        conversation = [(m.sender, m.performative.value) for m in state.messages]
        assert conversation == [
            ("classifier", "inform"),
            ("prioritizer", "request"),
            ("classifier", "agree"),
        ]
        assert state.classified_requirements[0].requirement_type is RequirementType.FUNCTIONAL
        [agree] = inbox_of(state.messages, "prioritizer", Performative.AGREE)
        assert isinstance(agree.content, RequestReply)
        assert isinstance(agree.content.revised, ClassificationReport)
