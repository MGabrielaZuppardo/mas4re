"""Agency conformance (docs/agency_conformance.md): what separates an agent from a workflow,
checked on the code and on scripted models. C5 and C7 need real runs and are not tested here.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.agentic.classifier import AgenticClassifier
from agents.agentic.prioritizer import AgenticPrioritizer
from agents.agentic.runtime import DEFAULT_MAX_STEPS
from domain.agent_actions import AgentEpisode
from domain.enums import RequirementType
from domain.messages import (
    ClassificationReport,
    Message,
    Performative,
    ReclassificationRequest,
)
from domain.models import ClassifiedRequirement, PipelineState, Requirement
from evaluation.agent_activity import summarize_episodes
from experiments.strategy import AgenticPipelineStrategy
from tests.helpers.scripted_llm import ScriptedChatModel, tool_call

_ROOT = Path(__file__).resolve().parents[2]
_AGENTIC = _ROOT / "agents" / "agentic"
_TEXT = "O sistema deve permitir que o usuário exporte relatórios"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _attribute_names(path: Path) -> set[str]:
    return {node.attr for node in ast.walk(_tree(path)) if isinstance(node, ast.Attribute)}


def _classifier(script: list) -> AgenticClassifier:
    with patch(
        "agents.agentic.classifier.build_llm", return_value=ScriptedChatModel(script=script)
    ):
        return AgenticClassifier(model="ollama/fake")


def _prioritizer(script: list) -> AgenticPrioritizer:
    with patch(
        "agents.agentic.prioritizer.build_llm", return_value=ScriptedChatModel(script=script)
    ):
        return AgenticPrioritizer(model="ollama/fake")


def _submit(category: str | None = "PE", confidence: float = 0.9):
    return tool_call(
        "submit_classification",
        requirement_type="NF" if category else "F",
        nfr_category=category,
        confidence=confidence,
        justification="ok",
    )


class TestC1NextStepIsTheModelsAction:
    def test_topologia_so_tem_entrada_agentes_e_verificacao_nunca_passos_internos(self):
        with (
            patch("agents.agentic.classifier.build_llm"),
            patch("agents.agentic.prioritizer.build_llm"),
        ):
            strategy = AgenticPipelineStrategy("ollama/a", "ollama/b")

        nodes = set(strategy._graph.get_graph().nodes)

        assert nodes == {"__start__", "classifier", "prioritizer", "cross_check", "__end__"}

    def test_o_laco_nao_conhece_nenhuma_ferramenta_nem_sequencia(self):
        source = (_AGENTIC / "runtime.py").read_text(encoding="utf-8")

        for name in (
            "lookup_nfr_taxonomy",
            "recall_similar_cases",
            "check_consistency",
            "backlog_overview",
            "request_reclassification",
            "submit_classification",
            "submit_priority",
        ):
            assert name not in source

    @pytest.mark.parametrize("module", ["classifier.py", "prioritizer.py"])
    def test_os_agentes_nunca_chamam_ferramentas_por_conta_propria(self, module):
        assert "invoke" not in _attribute_names(_AGENTIC / module)

    def test_o_mesmo_agente_percorre_caminhos_diferentes_conforme_o_modelo(self):
        direct = _classifier([_submit()])
        careful = _classifier(
            [
                tool_call("lookup_nfr_taxonomy", category_code="PE"),
                tool_call("check_consistency", requirement_type="NF", nfr_category="PE"),
                _submit(),
            ]
        )

        for agent in (direct, careful):
            agent.classify_batch([Requirement(id="r1", text=_TEXT)])

        assert direct._episodes[0].action_sequence != careful._episodes[0].action_sequence
        summary = summarize_episodes([*direct._episodes, *careful._episodes])["classifier"]
        assert summary["action_sequence_entropy_bits"] > 0


class TestC2TheAgentDecidesWhenToFinish:
    def test_quem_termina_e_o_agente_e_o_orcamento_so_impede_o_excesso(self):
        finishing = _classifier([_submit()])
        looping = _classifier(
            [tool_call("lookup_nfr_taxonomy", category_code="PE")] * DEFAULT_MAX_STEPS
        )

        finishing.classify_batch([Requirement(id="r1", text=_TEXT)])
        looping.classify_batch([Requirement(id="r1", text=_TEXT)])

        assert (
            summarize_episodes(finishing._episodes)["classifier"]["budget_exhaustion_rate"] == 0.0
        )
        assert summarize_episodes(looping._episodes)["classifier"]["budget_exhaustion_rate"] == 1.0

    def test_o_codigo_nunca_escreve_a_resposta_no_lugar_do_agente(self):
        looping = _classifier(
            [tool_call("lookup_nfr_taxonomy", category_code="PE")] * DEFAULT_MAX_STEPS
        )

        classified = looping.classify_batch([Requirement(id="r1", text=_TEXT)])

        assert classified == []


class TestC3OnlyTypedMessages:
    def test_priorizador_nao_le_a_saida_do_classificador_no_estado(self):
        attributes = _attribute_names(_AGENTIC / "prioritizer.py")

        assert "classified_requirements" not in attributes

    def test_priorizador_nao_importa_o_classificador(self):
        imports = {
            alias.name if isinstance(node, ast.Import) else node.module
            for node in ast.walk(_tree(_AGENTIC / "prioritizer.py"))
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in (node.names if isinstance(node, ast.Import) else [node])
        }

        assert not any(name and name.startswith("agents.agentic.classifier") for name in imports)

    def test_classificador_nao_le_a_saida_do_priorizador_no_estado(self):
        assert "prioritized_requirements" not in _attribute_names(_AGENTIC / "classifier.py")

    def test_o_priorizador_so_sabe_do_que_esta_na_mensagem(self):
        agent = _prioritizer([tool_call("submit_priority", priority="M", justification="ok")])
        state = PipelineState(
            messages=[
                Message(
                    sender="classifier",
                    receiver="prioritizer",
                    performative=Performative.INFORM,
                    requirement_id="r1",
                    content=ClassificationReport(
                        text=_TEXT,
                        requirement_type=RequirementType.NON_FUNCTIONAL,
                        nfr_category="US",
                        confidence=0.73,
                        justification="do relatório",
                    ),
                )
            ],
            classified_requirements=[
                ClassifiedRequirement(
                    id="r1",
                    text=_TEXT,
                    requirement_type=RequirementType.FUNCTIONAL,
                    confidence=0.11,
                )
            ],
        )

        agent.run(state)

        [item] = state.prioritized_requirements
        assert (item.nfr_category, item.confidence) == ("US", 0.73)


class TestC4TheAgentCanRefuse:
    @staticmethod
    def _answer(decision_call) -> Performative:
        classifier = _classifier([decision_call])
        state = PipelineState(
            raw_requirements=[Requirement(id="r1", text=_TEXT)],
            classified_requirements=[
                ClassifiedRequirement(
                    id="r1",
                    text=_TEXT,
                    requirement_type=RequirementType.NON_FUNCTIONAL,
                    nfr_category="US",
                    confidence=0.6,
                )
            ],
            messages=[
                Message(
                    sender="prioritizer",
                    receiver="classifier",
                    performative=Performative.REQUEST,
                    requirement_id="r1",
                    content=ReclassificationRequest(reason="parece funcional"),
                )
            ],
        )
        classifier.answer_requests(state)
        return state.messages[-1].performative

    def test_aceite_e_recusa_sao_ambos_possiveis_e_dependem_so_da_decisao_do_modelo(self):
        agree = tool_call(
            "reply_to_request",
            decision="agree",
            reason="sim",
            requirement_type="F",
            confidence=0.8,
        )
        refuse = tool_call("reply_to_request", decision="refuse", reason="não")

        assert {self._answer(agree), self._answer(refuse)} == {
            Performative.AGREE,
            Performative.REFUSE,
        }


class TestC6GoalsAndBeliefsAreInspectable:
    @staticmethod
    def _assert_visible(episode: AgentEpisode, first_calls: list) -> None:
        system, percept = first_calls[0][0].content, first_calls[0][1].content
        assert episode.goal.description in system
        assert episode.beliefs
        for belief in episode.beliefs:
            assert belief.content in system + percept

    def test_classificador(self):
        llm = ScriptedChatModel(script=[_submit()])
        with patch("agents.agentic.classifier.build_llm", return_value=llm):
            agent = AgenticClassifier(model="ollama/fake")

        agent.classify_batch([Requirement(id="r1", text=_TEXT)])

        self._assert_visible(agent._episodes[0], llm.seen)

    def test_priorizador(self):
        llm = ScriptedChatModel(
            script=[tool_call("submit_priority", priority="M", justification="ok")]
        )
        with patch("agents.agentic.prioritizer.build_llm", return_value=llm):
            agent = AgenticPrioritizer(model="ollama/fake")
        message = Message(
            sender="classifier",
            receiver="prioritizer",
            performative=Performative.INFORM,
            requirement_id="r1",
            content=ClassificationReport(
                text=_TEXT,
                requirement_type=RequirementType.NON_FUNCTIONAL,
                nfr_category="US",
                confidence=0.7,
                justification="do relatório",
            ),
        )
        state = PipelineState(messages=[message])

        agent.run(state)

        episode = AgentEpisode.model_validate(state.metrics["agent_episodes"][0])
        self._assert_visible(episode, llm.seen)

    def test_classificador_ao_responder_um_pedido(self):
        llm = ScriptedChatModel(
            script=[tool_call("reply_to_request", decision="refuse", reason="correto")]
        )
        with patch("agents.agentic.classifier.build_llm", return_value=llm):
            agent = AgenticClassifier(model="ollama/fake")
        state = PipelineState(
            classified_requirements=[
                ClassifiedRequirement(
                    id="r1",
                    text=_TEXT,
                    requirement_type=RequirementType.NON_FUNCTIONAL,
                    nfr_category="US",
                    confidence=0.6,
                )
            ],
            messages=[
                Message(
                    sender="prioritizer",
                    receiver="classifier",
                    performative=Performative.REQUEST,
                    requirement_id="r1",
                    content=ReclassificationRequest(reason="parece funcional"),
                )
            ],
        )

        agent.answer_requests(state)

        episode = AgentEpisode.model_validate(state.metrics["agent_episodes"][0])
        self._assert_visible(episode, llm.seen)
