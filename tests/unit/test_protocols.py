"""The coordination protocols P0/P2/P3 (ADR-013): same agents, different rules of conversation.

Everything runs on scripted models. The scripts are consumed in the order the agents act, so each
test reads as the story of one run.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.agentic.prioritizer import AgenticPrioritizer
from domain.agent_actions import ActionKind, AgentEpisode, EpisodePhase
from domain.enums import CoordinationProtocol, MoSCoWPriority, RequirementType
from domain.messages import (
    ClassificationReport,
    Message,
    Performative,
    PriorityReport,
    ReprioritizationRequest,
    RequestReply,
    inbox_of,
)
from domain.models import PipelineState, PrioritizedRequirement, Requirement
from evaluation.agent_activity import summarize_messages
from experiments.strategy import AGENTIC_STRATEGY_PROTOCOLS, AgenticPipelineStrategy
from pipeline.mediation import MEDIATOR
from tests.helpers.scripted_llm import ScriptedChatModel, tool_call

FIXED = CoordinationProtocol.FIXED
PEERS = CoordinationProtocol.PEERS
MEDIATED = CoordinationProtocol.MEDIATED

_TEXT = "O sistema deve permitir que o usuário exporte relatórios"


# ── scripted turns ─────────────────────────────────────────────────────────────────────


def _classify(requirement_type: str = "NF", category: str | None = "US", confidence: float = 0.6):
    return tool_call(
        "submit_classification",
        requirement_type=requirement_type,
        nfr_category=category,
        confidence=confidence,
        justification="classificação inicial",
    )


def _prioritize(priority: str):
    return tool_call("submit_priority", priority=priority, justification=f"decidi {priority}")


def _ask_classifier():
    return tool_call("request_reclassification", reason="parece funcional")


def _classifier_agrees():
    return tool_call(
        "reply_to_request",
        decision="agree",
        reason="tem razão",
        requirement_type="F",
        nfr_category=None,
        confidence=0.85,
        justification="é uma função do sistema",
    )


def _classifier_refuses():
    return tool_call("reply_to_request", decision="refuse", reason="a categoria está correta")


def _prioritizer_agrees(priority: str):
    return tool_call(
        "reply_to_request",
        decision="agree",
        reason="o argumento procede",
        priority=priority,
        justification=f"revisto para {priority}",
    )


def _prioritizer_refuses():
    return tool_call("reply_to_request", decision="refuse", reason="mantenho a decisão")


# ── builders ───────────────────────────────────────────────────────────────────────────


def _strategy(
    protocol: CoordinationProtocol,
    classifier_script: list,
    prioritizer_script: list,
    **kwargs,
) -> tuple[AgenticPipelineStrategy, ScriptedChatModel, ScriptedChatModel]:
    classifier_llm = ScriptedChatModel(script=classifier_script)
    prioritizer_llm = ScriptedChatModel(script=prioritizer_script)
    with (
        patch("agents.agentic.classifier.build_llm", return_value=classifier_llm),
        patch("agents.agentic.prioritizer.build_llm", return_value=prioritizer_llm),
    ):
        strategy = AgenticPipelineStrategy(
            "ollama/clf", "ollama/pri", max_steps=4, protocol=protocol, **kwargs
        )
    return strategy, classifier_llm, prioritizer_llm


def _requirements(count: int = 1) -> list[Requirement]:
    return [Requirement(id=f"r{i}", text=f"{_TEXT} {i}") for i in range(1, count + 1)]


def _prioritizer(
    script: list, protocol: CoordinationProtocol
) -> tuple[AgenticPrioritizer, ScriptedChatModel]:
    llm = ScriptedChatModel(script=script)
    with patch("agents.agentic.prioritizer.build_llm", return_value=llm):
        return AgenticPrioritizer(model="ollama/fake", protocol=protocol), llm


def _prioritized(priority: MoSCoWPriority, category: str | None = None) -> PrioritizedRequirement:
    return PrioritizedRequirement(
        id="r1",
        text=_TEXT,
        requirement_type=(
            RequirementType.NON_FUNCTIONAL if category else RequirementType.FUNCTIONAL
        ),
        nfr_category=category,
        confidence=0.9,
        priority=priority,
        priority_score=priority.score,
        priority_rank=1,
        priority_justification="decisão inicial",
    )


def _reprioritization_request(requirement_id: str = "r1") -> Message:
    return Message(
        sender=MEDIATOR,
        receiver="prioritizer",
        performative=Performative.REQUEST,
        requirement_id=requirement_id,
        content=ReprioritizationRequest(reason="Musts demais", evidence="100% do backlog é Must"),
    )


def _state_with_inform(
    requirement_type: RequirementType = RequirementType.NON_FUNCTIONAL,
    category: str | None = "US",
) -> PipelineState:
    return PipelineState(
        messages=[
            Message(
                sender="classifier",
                receiver="prioritizer",
                performative=Performative.INFORM,
                requirement_id="r1",
                content=ClassificationReport(
                    text=_TEXT,
                    requirement_type=requirement_type,
                    nfr_category=category,
                    confidence=0.6,
                    justification="classificação inicial",
                ),
            )
        ]
    )


def _conversation(state: PipelineState) -> list[tuple[str, str]]:
    return [(m.sender, m.performative.value) for m in state.messages]


def _episodes(state: PipelineState) -> list[AgentEpisode]:
    return [AgentEpisode.model_validate(e) for e in state.metrics["agent_episodes"]]


def _priority_of(state: PipelineState, requirement_id: str = "r1") -> MoSCoWPriority:
    item = next(p for p in state.prioritized_requirements if p.id == requirement_id)
    assert item.priority is not None
    return item.priority


# ── the strategy names the protocol ────────────────────────────────────────────────────


class TestStrategyNames:
    def test_cada_protocolo_tem_um_nome_de_estrategia_e_vice_versa(self):
        assert AGENTIC_STRATEGY_PROTOCOLS == {
            "pipeline_agentic": FIXED,
            "pipeline_agentic_p2": PEERS,
            "pipeline_agentic_p3": MEDIATED,
        }

    @pytest.mark.parametrize("protocol", [FIXED, PEERS, MEDIATED])
    def test_nome_e_protocolo_ficam_no_estado_da_execucao(self, protocol):
        strategy, *_ = _strategy(protocol, [_classify()], [_prioritize("S")])

        state = strategy.execute(_requirements())

        assert AGENTIC_STRATEGY_PROTOCOLS[strategy.name] is protocol
        assert state.metrics["protocol"] == protocol.value


# ── graph topology ─────────────────────────────────────────────────────────────────────


class TestTopology:
    @staticmethod
    def _nodes(protocol: CoordinationProtocol) -> set[str]:
        strategy, *_ = _strategy(protocol, [], [])
        return set(strategy._graph.get_graph().nodes) - {"__start__", "__end__"}

    def test_p0_e_o_grafo_fixo_sem_nenhum_no_de_interacao(self):
        assert self._nodes(FIXED) == {"classifier", "prioritizer", "cross_check"}

    def test_p2_acrescenta_as_vezes_de_resposta_dos_pares(self):
        assert self._nodes(PEERS) == {
            "classifier",
            "prioritizer",
            "classifier_answers",
            "prioritizer_reconsiders",
            "cross_check",
        }

    def test_p3_acrescenta_o_mediador_e_a_vez_de_resposta_do_priorizador(self):
        assert self._nodes(MEDIATED) == {
            "classifier",
            "prioritizer",
            "mediator",
            "prioritizer_answers",
            "cross_check",
        }

    @pytest.mark.parametrize("protocol", [PEERS, MEDIATED])
    def test_nenhum_protocolo_tem_laco_de_volta_a_um_no_anterior(self, protocol):
        strategy, *_ = _strategy(protocol, [], [])
        edges = strategy._graph.get_graph().edges

        assert all(not edge.conditional for edge in edges)
        assert len(edges) == len(self._nodes(protocol)) + 1


# ── P0: the agents have no channel back ────────────────────────────────────────────────


class TestFixedProtocol:
    def test_priorizador_nao_tem_como_pedir_nada(self):
        strategy, *_ = _strategy(FIXED, [_classify()], [_ask_classifier(), _prioritize("S")])

        state = strategy.execute(_requirements())

        [episode] = [e for e in _episodes(state) if e.agent == "prioritizer"]
        assert episode.actions[0].kind is ActionKind.INVALID
        assert "Unknown tool" in episode.actions[0].observation
        assert inbox_of(state.messages, "classifier") == []

    def test_o_prompt_so_apresenta_a_ferramenta_de_pedido_nos_protocolos_que_a_tem(self):
        shown = {}
        for protocol in (FIXED, PEERS, MEDIATED):
            agent, llm = _prioritizer([_prioritize("S")], protocol)
            agent.run(_state_with_inform())
            shown[protocol] = "request_reclassification" in llm.seen[0][0].content

        assert shown == {FIXED: False, PEERS: True, MEDIATED: False}

    def test_as_operacoes_sociais_so_existem_no_protocolo_que_as_usa(self):
        fixed, _ = _prioritizer([], FIXED)
        peers, _ = _prioritizer([], PEERS)
        mediated, _ = _prioritizer([], MEDIATED)

        with pytest.raises(ValueError, match="PEERS"):
            fixed.reconsider(PipelineState())
        with pytest.raises(ValueError, match="MEDIATED"):
            peers.answer_requests(PipelineState())
        with pytest.raises(ValueError, match="PEERS"):
            mediated.reconsider(PipelineState())
        with pytest.raises(ValueError, match="MEDIATED"):
            fixed.answer_requests(PipelineState())


# ── P2: the prioritizer asks, the classifier decides, the prioritizer decides again ────


class TestPeersProtocol:
    def test_pedido_aceito_corrige_a_classificacao_e_o_priorizador_decide_de_novo(self):
        strategy, *_ = _strategy(
            PEERS,
            [_classify(), _classifier_agrees()],
            [_ask_classifier(), _prioritize("S"), _prioritize("C")],
        )

        state = strategy.execute(_requirements())

        assert _conversation(state) == [
            ("classifier", "inform"),
            ("prioritizer", "request"),
            ("classifier", "agree"),
            ("prioritizer", "inform"),
        ]
        [item] = state.prioritized_requirements
        assert item.requirement_type is RequirementType.FUNCTIONAL and item.nfr_category is None
        assert item.confidence == 0.85
        assert item.priority is MoSCoWPriority.COULD_HAVE
        assert state.classified_requirements[0].requirement_type is RequirementType.FUNCTIONAL

    def test_pedido_recusado_mantem_a_classificacao_e_a_decisao_final_e_do_priorizador(self):
        strategy, *_ = _strategy(
            PEERS,
            [_classify(), _classifier_refuses()],
            [_ask_classifier(), _prioritize("S"), _prioritize("M")],
        )

        state = strategy.execute(_requirements())

        assert _conversation(state)[2] == ("classifier", "refuse")
        [item] = state.prioritized_requirements
        assert (item.requirement_type, item.nfr_category) == (RequirementType.NON_FUNCTIONAL, "US")
        assert item.priority is MoSCoWPriority.MUST_HAVE

    def test_a_resposta_do_par_chega_ao_priorizador_na_percepcao_da_reconsideracao(self):
        strategy, _, prioritizer_llm = _strategy(
            PEERS,
            [_classify(), _classifier_refuses()],
            [_ask_classifier(), _prioritize("S"), _prioritize("S")],
        )

        strategy.execute(_requirements())

        system, percept = prioritizer_llm.seen[2][0].content, prioritizer_llm.seen[2][1].content
        assert "a categoria está correta" in percept and "decidi S" in percept
        assert "request_reclassification" not in system

    def test_na_reconsideracao_nao_ha_como_pedir_de_novo_o_dialogo_tem_uma_rodada(self):
        strategy, *_ = _strategy(
            PEERS,
            [_classify(), _classifier_agrees()],
            [_ask_classifier(), _prioritize("S"), _ask_classifier(), _prioritize("C")],
        )

        state = strategy.execute(_requirements())

        prioritizer_requests = [
            m
            for m in state.messages
            if m.sender == "prioritizer" and m.performative is Performative.REQUEST
        ]
        assert len(prioritizer_requests) == 1
        [reconsideration] = [e for e in _episodes(state) if e.phase is EpisodePhase.RECONSIDERATION]
        assert reconsideration.actions[0].kind is ActionKind.INVALID

    def test_sem_pedido_nao_ha_episodio_extra_e_o_modelo_nao_e_chamado(self):
        strategy, classifier_llm, prioritizer_llm = _strategy(
            PEERS, [_classify()], [_prioritize("S")]
        )

        state = strategy.execute(_requirements())

        assert (classifier_llm.calls, prioritizer_llm.calls) == (1, 1)
        assert {e.phase for e in _episodes(state)} == {EpisodePhase.WORK}

    def test_cada_turno_do_protocolo_registra_a_sua_fase(self):
        strategy, *_ = _strategy(
            PEERS,
            [_classify(), _classifier_agrees()],
            [_ask_classifier(), _prioritize("S"), _prioritize("C")],
        )

        state = strategy.execute(_requirements())

        assert [(e.agent, e.phase) for e in _episodes(state)] == [
            ("classifier", EpisodePhase.WORK),
            ("prioritizer", EpisodePhase.WORK),
            ("classifier", EpisodePhase.REPLY),
            ("prioritizer", EpisodePhase.RECONSIDERATION),
        ]

    def test_pedido_que_o_par_nao_conseguiu_responder_nao_gera_reconsideracao(self):
        strategy, _, prioritizer_llm = _strategy(
            PEERS,
            [_classify(), *[tool_call("lookup_nfr_taxonomy", category_code="US")] * 4],
            [_ask_classifier(), _prioritize("S")],
        )

        state = strategy.execute(_requirements())

        assert _conversation(state)[2] == ("classifier", "failure")
        assert prioritizer_llm.calls == 2
        assert _priority_of(state) is MoSCoWPriority.SHOULD_HAVE

    def test_reconsiderar_de_novo_nao_repete_o_que_ja_foi_reconsiderado(self):
        strategy, _, prioritizer_llm = _strategy(
            PEERS,
            [_classify(), _classifier_agrees()],
            [_ask_classifier(), _prioritize("S"), _prioritize("C")],
        )
        state = strategy.execute(_requirements())
        calls = prioritizer_llm.calls

        strategy._prioritizer.reconsider(state)

        assert prioritizer_llm.calls == calls

    def test_o_resumo_de_interacao_mostra_o_aceite(self):
        strategy, *_ = _strategy(
            PEERS,
            [_classify(), _classifier_agrees()],
            [_ask_classifier(), _prioritize("S"), _prioritize("C")],
        )

        summary = summarize_messages(strategy.execute(_requirements()).messages)

        assert summary["requests"] == 1 and summary["requests_by_sender"] == {"prioritizer": 1}
        assert (summary["agree"], summary["refuse"], summary["unanswered"]) == (1, 0, 0)
        assert summary["agree_rate"] == 1.0 and summary["refuse_rate"] == 0.0


# ── P3: the mediator asks, the prioritizer decides ─────────────────────────────────────


class TestMediatedProtocol:
    def test_inflacao_de_must_aceita_o_priorizador_revisa_e_responde_ao_mediador(self):
        strategy, *_ = _strategy(
            MEDIATED, [_classify("F", None)], [_prioritize("M"), _prioritizer_agrees("S")]
        )

        state = strategy.execute(_requirements())

        assert _conversation(state) == [
            ("classifier", "inform"),
            (MEDIATOR, "request"),
            ("prioritizer", "agree"),
        ]
        [request] = inbox_of(state.messages, "prioritizer", Performative.REQUEST)
        [reply] = inbox_of(state.messages, MEDIATOR, Performative.AGREE)
        assert reply.in_reply_to == request.id and isinstance(reply.content, RequestReply)
        assert isinstance(reply.content.revised_priority, PriorityReport)
        assert reply.content.revised_priority.priority is MoSCoWPriority.SHOULD_HAVE
        assert _priority_of(state) is MoSCoWPriority.SHOULD_HAVE
        assert state.metrics["mediator"] == {
            "conflicts": 0,
            "must_inflation": 1,
            "requests": 1,
            "capped": False,
            "must_limit": 0.6,
            "must_share_before": 1.0,
            "must_share_after": 0.0,
        }

    def test_recusa_mantem_a_prioridade_e_fica_registrada_com_o_motivo(self):
        strategy, *_ = _strategy(
            MEDIATED, [_classify("F", None)], [_prioritize("M"), _prioritizer_refuses()]
        )

        state = strategy.execute(_requirements())

        [reply] = inbox_of(state.messages, MEDIATOR, Performative.REFUSE)
        assert isinstance(reply.content, RequestReply)
        assert reply.content.reason == "mantenho a decisão"
        assert reply.content.revised_priority is None
        assert _priority_of(state) is MoSCoWPriority.MUST_HAVE
        assert state.metrics["mediator"]["must_share_after"] == 1.0

    def test_conflito_de_categoria_critica_com_prioridade_baixa_chega_como_pedido(self):
        strategy, *_ = _strategy(
            MEDIATED, [_classify("NF", "SE")], [_prioritize("C"), _prioritizer_agrees("M")]
        )

        state = strategy.execute(_requirements())

        [request] = inbox_of(state.messages, "prioritizer", Performative.REQUEST)
        assert isinstance(request.content, ReprioritizationRequest)
        assert "'SE'" in request.content.evidence
        assert state.metrics["mediator"]["conflicts"] == 1
        assert _priority_of(state) is MoSCoWPriority.MUST_HAVE

    def test_o_priorizador_ve_o_pedido_e_a_evidencia_na_percepcao(self):
        strategy, _, prioritizer_llm = _strategy(
            MEDIATED, [_classify("F", None)], [_prioritize("M"), _prioritizer_refuses()]
        )

        strategy.execute(_requirements())

        percept = prioritizer_llm.seen[1][1].content
        assert "100%" in percept and "decidi M" in percept

    def test_sem_achados_nao_ha_pedido_nem_episodio_extra(self):
        strategy, _, prioritizer_llm = _strategy(
            MEDIATED, [_classify("F", None)], [_prioritize("S")]
        )

        state = strategy.execute(_requirements())

        assert prioritizer_llm.calls == 1
        assert inbox_of(state.messages, "prioritizer", Performative.REQUEST) == []
        assert state.metrics["mediator"]["requests"] == 0
        assert {e.phase for e in _episodes(state)} == {EpisodePhase.WORK}

    def test_o_teto_de_pedidos_limita_o_dialogo_e_o_mediador_declara_o_corte(self):
        strategy, *_ = _strategy(
            MEDIATED,
            [_classify("NF", "SE"), _classify("NF", "SE")],
            [_prioritize("C"), _prioritize("C"), _prioritizer_agrees("M")],
            max_requests=1,
        )

        state = strategy.execute(_requirements(2))

        assert state.metrics["mediator"]["conflicts"] == 2
        assert state.metrics["mediator"]["requests"] == 1
        assert state.metrics["mediator"]["capped"] is True
        assert len(inbox_of(state.messages, "prioritizer", Performative.REQUEST)) == 1
        assert {p.id: p.priority for p in state.prioritized_requirements} == {
            "r1": MoSCoWPriority.MUST_HAVE,
            "r2": MoSCoWPriority.COULD_HAVE,
        }

    def test_aceitar_sem_a_prioridade_revista_e_acao_invalida_e_o_agente_corrige(self):
        agree_without_priority = tool_call("reply_to_request", decision="agree", reason="sim")
        strategy, *_ = _strategy(
            MEDIATED,
            [_classify("F", None)],
            [_prioritize("M"), agree_without_priority, _prioritizer_agrees("S")],
        )

        state = strategy.execute(_requirements())

        [reply_episode] = [e for e in _episodes(state) if e.phase is EpisodePhase.REPLY]
        assert reply_episode.actions[0].kind is ActionKind.INVALID
        assert "priority" in reply_episode.actions[0].observation
        assert _priority_of(state) is MoSCoWPriority.SHOULD_HAVE

    def test_sem_resposta_no_orcamento_o_priorizador_responde_failure_e_nada_muda(self):
        strategy, *_ = _strategy(
            MEDIATED,
            [_classify("F", None)],
            [_prioritize("M"), *[tool_call("backlog_overview")] * 4],
        )

        state = strategy.execute(_requirements())

        [reply] = inbox_of(state.messages, MEDIATOR)
        assert reply.performative is Performative.FAILURE
        assert _priority_of(state) is MoSCoWPriority.MUST_HAVE

    def test_o_turno_de_resposta_registra_a_fase_e_a_acao_terminal(self):
        strategy, *_ = _strategy(
            MEDIATED, [_classify("F", None)], [_prioritize("M"), _prioritizer_refuses()]
        )

        state = strategy.execute(_requirements())

        [reply_episode] = [e for e in _episodes(state) if e.phase is EpisodePhase.REPLY]
        assert reply_episode.agent == "prioritizer"
        assert reply_episode.action_sequence == ("reply_to_request",)

    def test_o_resumo_de_interacao_conta_os_pedidos_do_mediador(self):
        strategy, *_ = _strategy(
            MEDIATED, [_classify("F", None)], [_prioritize("M"), _prioritizer_refuses()]
        )

        summary = summarize_messages(strategy.execute(_requirements()).messages)

        assert summary["requests_by_sender"] == {MEDIATOR: 1}
        assert (summary["agree"], summary["refuse"]) == (0, 1)
        assert summary["refuse_rate"] == 1.0 and summary["agree_rate"] == 0.0

    def test_a_decisao_final_depende_do_que_o_agente_conclui_e_nao_de_o_mediador_pedir(self):
        agreeing, *_ = _strategy(
            MEDIATED, [_classify("F", None)], [_prioritize("M"), _prioritizer_agrees("S")]
        )
        refusing, *_ = _strategy(
            MEDIATED, [_classify("F", None)], [_prioritize("M"), _prioritizer_refuses()]
        )

        first, second = agreeing.execute(_requirements()), refusing.execute(_requirements())

        assert {_priority_of(first), _priority_of(second)} == {
            MoSCoWPriority.SHOULD_HAVE,
            MoSCoWPriority.MUST_HAVE,
        }


# ── the prioritizer answering requests directly (without the graph) ────────────────────


class TestPrioritizerAnswers:
    @staticmethod
    def _state(*requests: Message) -> PipelineState:
        return PipelineState(
            prioritized_requirements=[_prioritized(MoSCoWPriority.MUST_HAVE)],
            messages=list(requests),
        )

    def test_pedido_para_requisito_desconhecido_recebe_failure_sem_gastar_o_modelo(self):
        agent, llm = _prioritizer([], MEDIATED)
        state = self._state(_reprioritization_request("inexistente"))

        agent.answer_requests(state)

        assert llm.calls == 0
        assert state.messages[-1].performative is Performative.FAILURE

    def test_pedido_ja_respondido_nao_e_respondido_de_novo(self):
        agent, llm = _prioritizer([_prioritizer_refuses()], MEDIATED)
        state = self._state(_reprioritization_request())

        agent.answer_requests(state)
        agent.answer_requests(state)

        assert llm.calls == 1
        assert len(inbox_of(state.messages, MEDIATOR, Performative.REFUSE)) == 1

    def test_aceite_usa_o_score_que_o_agente_informou_e_a_justificativa_nova(self):
        agent, _ = _prioritizer(
            [
                tool_call(
                    "reply_to_request",
                    decision="agree",
                    reason="ok",
                    priority="S",
                    priority_score=0.6,
                    justification="rebaixado",
                )
            ],
            MEDIATED,
        )
        state = self._state(_reprioritization_request())

        agent.answer_requests(state)

        [item] = state.prioritized_requirements
        assert (item.priority, item.priority_score) == (MoSCoWPriority.SHOULD_HAVE, 0.6)
        assert item.priority_justification == "rebaixado"


class TestMediatorLimit:
    @staticmethod
    def _backlog(musts: int, total: int):
        from domain.enums import Lang, MoSCoWPriority, RequirementType
        from domain.models import PrioritizedRequirement

        return [
            PrioritizedRequirement(
                id=f"r{i}",
                text=f"Requisito {i}",
                requirement_type=RequirementType.FUNCTIONAL,
                confidence=0.9,
                priority=MoSCoWPriority.MUST_HAVE if i < musts else MoSCoWPriority.SHOULD_HAVE,
                priority_score=1.0 - i / 100,
                priority_rank=i + 1,
            )
            for i in range(total)
        ], Lang.PT

    def test_limite_padrao_e_60_por_cento(self):
        from pipeline.mediation import MUST_SHARE_LIMIT, mediate

        items, lang = self._backlog(musts=5, total=10)

        assert MUST_SHARE_LIMIT == 0.60
        assert mediate(items, lang).must_inflation == 0

    def test_limite_menor_gera_pedidos_so_para_o_excesso(self):
        from pipeline.mediation import mediate

        items, lang = self._backlog(musts=5, total=10)

        report = mediate(items, lang, must_limit=0.40)

        assert report.must_inflation == 1 and len(report.findings) == 1

    def test_mediador_registra_o_limite_usado(self):
        from domain.models import PipelineState
        from pipeline.nodes.protocol_nodes import mediator_node

        items, lang = self._backlog(musts=5, total=10)
        state = PipelineState(raw_requirements=[], prioritized_requirements=items)

        mediator_node(state, lang, max_requests=25, must_limit=0.30)

        assert state.metrics["mediator"]["must_limit"] == 0.30
        assert state.metrics["mediator"]["must_inflation"] == 2
