from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.enums import RequirementType
from domain.messages import (
    ClassificationReport,
    Message,
    Performative,
    ReclassificationRequest,
    RequestReply,
    inbox_of,
    latest_per_requirement,
)
from domain.models import PipelineState


def _report(**overrides) -> ClassificationReport:
    fields = {
        "text": "O sistema deve responder em 200ms.",
        "requirement_type": RequirementType.NON_FUNCTIONAL,
        "nfr_category": "PE",
        "confidence": 0.9,
        "justification": "desempenho",
    }
    return ClassificationReport(**{**fields, **overrides})


def _inform(requirement_id: str, receiver: str = "prioritizer", **report) -> Message:
    return Message(
        sender="classifier",
        receiver=receiver,
        performative=Performative.INFORM,
        requirement_id=requirement_id,
        content=_report(**report),
    )


class TestMessage:
    def test_o_conteudo_volta_com_o_tipo_certo_depois_de_serializar(self):
        original = Message(
            sender="prioritizer",
            receiver="classifier",
            performative=Performative.REQUEST,
            requirement_id="r1",
            content=ReclassificationRequest(reason="parece funcional"),
        )

        restored = Message.model_validate(original.model_dump(mode="json"))

        assert isinstance(restored.content, ReclassificationRequest)
        assert restored.content.reason == "parece funcional"

    def test_resposta_carrega_o_relatorio_revisado(self):
        reply = Message(
            sender="classifier",
            receiver="prioritizer",
            performative=Performative.AGREE,
            requirement_id="r1",
            content=RequestReply(reason="tem razão", revised=_report(nfr_category="US")),
            in_reply_to="abc",
        )

        restored = Message.model_validate(reply.model_dump(mode="json"))

        assert isinstance(restored.content, RequestReply)
        assert (
            restored.content.revised is not None and restored.content.revised.nfr_category == "US"
        )
        assert restored.in_reply_to == "abc"

    def test_conteudo_de_tipo_desconhecido_e_recusado(self):
        with pytest.raises(ValidationError):
            Message.model_validate(
                {
                    "sender": "a",
                    "receiver": "b",
                    "performative": "inform",
                    "requirement_id": "r1",
                    "content": {"kind": "outro", "x": 1},
                }
            )

    def test_confianca_fora_do_intervalo_e_recusada(self):
        with pytest.raises(ValidationError):
            _report(confidence=1.4)

    def test_cada_mensagem_recebe_um_id_proprio(self):
        assert _inform("r1").id != _inform("r1").id


class TestInbox:
    def test_filtra_por_destinatario_e_performativa(self):
        request = Message(
            sender="prioritizer",
            receiver="classifier",
            performative=Performative.REQUEST,
            requirement_id="r1",
            content=ReclassificationRequest(reason="x"),
        )
        messages = [_inform("r1"), request, _inform("r2", receiver="classifier")]

        assert [m.requirement_id for m in inbox_of(messages, "prioritizer")] == ["r1"]
        assert inbox_of(messages, "classifier", Performative.REQUEST) == [request]
        assert inbox_of(messages, "ninguem") == []

    def test_estado_do_pipeline_guarda_as_mensagens(self):
        state = PipelineState(messages=[_inform("r1")])

        restored = PipelineState.model_validate(state.model_dump(mode="json"))

        assert restored.messages[0].requirement_id == "r1"


class TestLatestPerRequirement:
    def test_a_mensagem_mais_recente_substitui_a_anterior(self):
        first, second = _inform("r1", confidence=0.6), _inform("r1", confidence=0.95)

        result = latest_per_requirement([first, _inform("r2"), second])

        assert [m.requirement_id for m in result] == ["r1", "r2"]
        assert result[0] is second

    def test_sem_mensagens_devolve_lista_vazia(self):
        assert latest_per_requirement([]) == []
