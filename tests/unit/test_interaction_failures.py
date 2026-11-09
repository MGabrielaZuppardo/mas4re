"""Failures of the exchange between agents, read from the messages."""

from __future__ import annotations

from domain.failures import FailureMode
from domain.messages import (
    Message,
    Performative,
    PriorityReport,
    ReclassificationRequest,
    ReprioritizationRequest,
    RequestReply,
)
from evaluation.interaction_failures import (
    detect_interaction_failures,
    summarize_interaction_failures,
)


def _mediator_request(requirement_id="r1"):
    return Message(
        sender="mediator",
        receiver="prioritizer",
        performative=Performative.REQUEST,
        requirement_id=requirement_id,
        content=ReprioritizationRequest(reason="why", evidence="e"),
    )


def _peer_request(requirement_id="r1"):
    return Message(
        sender="prioritizer",
        receiver="classifier",
        performative=Performative.REQUEST,
        requirement_id=requirement_id,
        content=ReclassificationRequest(reason="why"),
    )


def _reply(request, sender, performative):
    return Message(
        sender=sender,
        receiver=request.sender,
        performative=performative,
        requirement_id=request.requirement_id,
        content=RequestReply(reason="r"),
        in_reply_to=request.id,
    )


def _acted_on(answer):
    return Message(
        sender="prioritizer",
        receiver="classifier",
        performative=Performative.INFORM,
        requirement_id=answer.requirement_id,
        content=PriorityReport(priority="M", priority_score=1.0),
        in_reply_to=answer.id,
    )


def _decision(initial, final, requirement_id="r1"):
    return {"id": requirement_id, "initial_priority": initial, "final_priority": final}


def _modes(messages, decisions=()):
    return [f.mode for f in detect_interaction_failures(messages, decisions)]


class TestUnanswered:
    def test_pedido_sem_resposta(self):
        assert _modes([_mediator_request()]) == [FailureMode.INTERACTION_UNANSWERED]

    def test_agente_que_responde_failure_nao_respondeu(self):
        request = _mediator_request()

        modes = _modes([request, _reply(request, "prioritizer", Performative.FAILURE)])

        assert modes == [FailureMode.INTERACTION_UNANSWERED]

    def test_resposta_do_classificador_sem_reconsideracao_foi_ignorada(self):
        request = _peer_request()
        answer = _reply(request, "classifier", Performative.REFUSE)

        assert _modes([request, answer]) == [FailureMode.INTERACTION_UNANSWERED]

    def test_resposta_seguida_de_reconsideracao_nao_e_falha(self):
        request = _peer_request()
        answer = _reply(request, "classifier", Performative.AGREE)

        assert _modes([request, answer, _acted_on(answer)]) == []


class TestInconsistent:
    def test_concordou_mas_a_prioridade_nao_mudou(self):
        request = _mediator_request()
        messages = [request, _reply(request, "prioritizer", Performative.AGREE)]

        assert _modes(messages, [_decision("M", "M")]) == [FailureMode.INTERACTION_INCONSISTENT]

    def test_recusou_mas_a_prioridade_mudou(self):
        request = _mediator_request()
        messages = [request, _reply(request, "prioritizer", Performative.REFUSE)]

        assert _modes(messages, [_decision("M", "S")]) == [FailureMode.INTERACTION_INCONSISTENT]

    def test_concordou_e_mudou_ou_recusou_e_manteve_e_coerente(self):
        agree = _mediator_request("r1")
        refuse = _mediator_request("r2")
        messages = [
            agree,
            _reply(agree, "prioritizer", Performative.AGREE),
            refuse,
            _reply(refuse, "prioritizer", Performative.REFUSE),
        ]
        decisions = [_decision("M", "S", "r1"), _decision("M", "M", "r2")]

        assert _modes(messages, decisions) == []

    def test_sem_decisoes_registradas_nao_julga_coerencia(self):
        request = _mediator_request()

        assert _modes([request, _reply(request, "prioritizer", Performative.AGREE)]) == []


class TestSummary:
    def test_resumo_conta_por_modo(self):
        summary = summarize_interaction_failures([_mediator_request("a"), _mediator_request("b")])

        assert summary["total"] == 2
        assert summary["by_mode"] == {"interaction_unanswered": 2}
        assert summary["examples"][0]["requirement_id"] == "a"

    def test_execucao_sem_mensagens_nao_tem_falhas(self):
        assert summarize_interaction_failures([])["total"] == 0
