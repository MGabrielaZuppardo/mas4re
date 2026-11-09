"""The architecture of each protocol as data, and the check that a run obeyed it."""

from __future__ import annotations

import json

import pytest

from agents.agentic.classifier import AgenticClassifier
from agents.agentic.prioritizer import AgenticPrioritizer
from domain.enums import CoordinationProtocol, MoSCoWPriority, RequirementType
from domain.messages import (
    ClassificationReport,
    Message,
    Performative,
    PriorityReport,
    ReclassificationRequest,
    ReprioritizationRequest,
    RequestReply,
)
from evaluation.protocol_conformance import (
    OUTSIDE_CHANNEL,
    REPLY_FROM_WRONG_AGENT,
    REPLY_TO_NOTHING,
    SELF_MESSAGE,
    TOO_MANY_REQUESTS,
    UNKNOWN_SENDER,
    check_conformance,
    summarize_conformance,
)
from pipeline.mediation import MEDIATOR
from pipeline.protocol_spec import CLASSIFIER, PRIORITIZER, PROTOCOL_SPECS

P0 = PROTOCOL_SPECS[CoordinationProtocol.FIXED]
P2 = PROTOCOL_SPECS[CoordinationProtocol.PEERS]
P3 = PROTOCOL_SPECS[CoordinationProtocol.MEDIATED]


def _report(requirement_id="r1"):
    return Message(
        sender=CLASSIFIER,
        receiver=PRIORITIZER,
        performative=Performative.INFORM,
        requirement_id=requirement_id,
        content=ClassificationReport(
            text="t", requirement_type=RequirementType.FUNCTIONAL, confidence=0.9
        ),
    )


def _ask_classifier(requirement_id="r1"):
    return Message(
        sender=PRIORITIZER,
        receiver=CLASSIFIER,
        performative=Performative.REQUEST,
        requirement_id=requirement_id,
        content=ReclassificationRequest(reason="why"),
    )


def _mediator_asks(requirement_id="r1"):
    return Message(
        sender=MEDIATOR,
        receiver=PRIORITIZER,
        performative=Performative.REQUEST,
        requirement_id=requirement_id,
        content=ReprioritizationRequest(reason="why", evidence="e"),
    )


def _answer(request, sender, performative=Performative.REFUSE):
    return Message(
        sender=sender,
        receiver=request.sender,
        performative=performative,
        requirement_id=request.requirement_id,
        content=RequestReply(reason="no"),
        in_reply_to=request.id,
    )


def _priority_report(answer):
    return Message(
        sender=PRIORITIZER,
        receiver=CLASSIFIER,
        performative=Performative.INFORM,
        requirement_id=answer.requirement_id,
        content=PriorityReport(priority=MoSCoWPriority.MUST_HAVE, priority_score=1.0),
        in_reply_to=answer.id,
    )


class TestProtocolSpec:
    def test_nomes_dos_papeis_sao_os_dos_agentes(self):
        assert CLASSIFIER == AgenticClassifier.name
        assert PRIORITIZER == AgenticPrioritizer.name

    def test_p0_so_tem_o_canal_de_repasse(self):
        assert len(P0.channels) == 1
        assert not P0.allows(
            PRIORITIZER, CLASSIFIER, Performative.REQUEST, "reclassification_request"
        )

    def test_p2_deixa_o_priorizador_pedir_ao_classificador(self):
        assert P2.allows(PRIORITIZER, CLASSIFIER, Performative.REQUEST, "reclassification_request")
        assert not P2.allows(
            MEDIATOR, PRIORITIZER, Performative.REQUEST, "reprioritization_request"
        )

    def test_p3_inclui_o_mediador_como_papel_e_canal(self):
        assert MEDIATOR in P3.roles
        assert P3.allows(MEDIATOR, PRIORITIZER, Performative.REQUEST, "reprioritization_request")

    def test_descricao_e_serializavel_para_o_manifest(self):
        description = json.loads(json.dumps(P3.describe()))

        assert description["protocol"] == "p3"
        assert any(
            c["from"] == "mediator" and c["to"] == "prioritizer" for c in description["channels"]
        )


class TestConformantRuns:
    def test_p0_so_com_repasse(self):
        assert check_conformance(P0, [_report("r1"), _report("r2")]) == []

    def test_p2_com_pedido_resposta_e_relatorio_de_prioridade(self):
        request = _ask_classifier()
        answer = _answer(request, CLASSIFIER)
        messages = [_report(), request, answer, _priority_report(answer)]

        assert check_conformance(P2, messages) == []

    def test_p3_com_pedido_do_mediador_e_resposta(self):
        request = _mediator_asks()

        assert check_conformance(P3, [_report(), request, _answer(request, PRIORITIZER)]) == []

    def test_sem_mensagens_nao_ha_violacao(self):
        assert check_conformance(P2, []) == []


class TestViolations:
    @staticmethod
    def _rules(spec, messages):
        return [v.rule for v in check_conformance(spec, messages)]

    def test_pedido_no_p0_esta_fora_do_canal(self):
        assert OUTSIDE_CHANNEL in self._rules(P0, [_report(), _ask_classifier()])

    def test_mediador_nao_participa_do_p2(self):
        assert UNKNOWN_SENDER in self._rules(P2, [_mediator_asks()])

    def test_resposta_a_um_pedido_inexistente(self):
        ghost = _ask_classifier()
        answer = _answer(ghost, CLASSIFIER)

        assert REPLY_TO_NOTHING in self._rules(P2, [answer])

    def test_resposta_de_quem_nao_recebeu_o_pedido(self):
        request = _ask_classifier()
        wrong = Message(
            sender=PRIORITIZER,
            receiver=PRIORITIZER,
            performative=Performative.REFUSE,
            requirement_id="r1",
            content=RequestReply(reason="x"),
            in_reply_to=request.id,
        )

        rules = self._rules(P2, [request, wrong])

        assert REPLY_FROM_WRONG_AGENT in rules and SELF_MESSAGE in rules

    def test_participante_que_o_protocolo_nao_tem(self):
        stranger = _report().model_copy(update={"sender": "auditor"})

        assert UNKNOWN_SENDER in self._rules(P2, [stranger])

    def test_dois_pedidos_para_o_mesmo_requisito(self):
        rules = self._rules(P2, [_ask_classifier(), _ask_classifier()])

        assert TOO_MANY_REQUESTS in rules

    def test_pedidos_para_requisitos_diferentes_nao_somam(self):
        assert check_conformance(P2, [_ask_classifier("r1"), _ask_classifier("r2")]) == []

    def test_resumo_conta_por_regra_e_limita_os_exemplos(self):
        messages = [_ask_classifier(f"r{i}") for i in range(30)]

        summary = summarize_conformance(P0, messages)

        assert summary["violations"] == 30
        assert summary["by_rule"] == {OUTSIDE_CHANNEL: 30}
        assert len(summary["examples"]) == 20
        assert summary["messages_checked"] == 30

    @pytest.mark.parametrize("spec", [P0, P2, P3])
    def test_resumo_limpo(self, spec):
        summary = summarize_conformance(spec, [_report()])

        assert summary["violations"] == 0 and summary["examples"] == []
