from unittest.mock import MagicMock, patch

import pytest

from agents.baseline import BaselineAgent
from domain.enums import Lang, MoSCoWPriority, RequirementType
from domain.models import PipelineState, Requirement


@pytest.fixture
def agent():
    with patch("agents.baseline.build_llm"):
        return BaselineAgent(model="ollama/qwen2.5:7b")


@pytest.fixture
def agent_with_categories():
    with patch("agents.baseline.build_llm"):
        return BaselineAgent(
            model="ollama/qwen2.5:7b",
            nfr_categories=[("SE", "Security"), ("PE", "Performance")],
        )


@pytest.fixture
def sample_requirement():
    return Requirement(id="req-01", text="O sistema deve autenticar usuários via OAuth2.")


def _mock_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.content = content
    return mock
class TestBaselineAgentParsing:

    def test_parse_funcional(self, agent):
        content = (
            '{"requirement_type": "F", "nfr_category": null, "confidence": 0.92,'
            ' "classification_justification": "Descreve ação do sistema.",'
            ' "priority": "M", "priority_score": 0.95, "priority_rank": 1,'
            ' "priority_justification": "Fluxo crítico."}'
        )
        output = agent._parse_response(content, "req-01")
        assert output.requirement_type == RequirementType.FUNCTIONAL
        assert output.nfr_category is None
        assert output.confidence == 0.92
        assert output.priority == MoSCoWPriority.MUST_HAVE
        assert output.priority_score == 0.95

    def test_parse_nfr_com_categoria_taxonomica(self, agent):
        content = (
            '{"requirement_type": "NF", "nfr_category": "SE", "confidence": 0.88,'
            ' "classification_justification": "Requisito de segurança.",'
            ' "priority": "S", "priority_score": 0.75, "priority_rank": 1,'
            ' "priority_justification": "Alta importância."}'
        )
        output = agent._parse_response(content, "req-02")
        assert output.requirement_type == RequirementType.NON_FUNCTIONAL
        assert output.nfr_category == "SE"
        assert output.priority == MoSCoWPriority.SHOULD_HAVE

    def test_parse_nfr_categoria_livre(self, agent):
        """Dataset agnóstico: aceita categoria como string livre."""
        content = (
            '{"requirement_type": "NF", "nfr_category": "desempenho", "confidence": 0.80,'
            ' "classification_justification": "Qualidade de desempenho.",'
            ' "priority": "S", "priority_score": 0.75, "priority_rank": 1,'
            ' "priority_justification": "Importante."}'
        )
        output = agent._parse_response(content, "req-03")
        assert output.nfr_category == "desempenho"

    def test_parse_json_invalido_retorna_fallback(self, agent):
        output = agent._parse_response("resposta inválida", "req-04")
        assert output.requirement_type == RequirementType.FUNCTIONAL
        assert output.confidence == 0.0
        assert output.priority == MoSCoWPriority.COULD_HAVE
        assert "Parse falhou" in output.classification_justification

    def test_parse_json_com_markdown(self, agent):
        content = (
            "```json\n"
            '{"requirement_type": "F", "nfr_category": null, "confidence": 0.9,'
            ' "classification_justification": "ok", "priority": "C",'
            ' "priority_score": 0.5, "priority_rank": 1, "priority_justification": "ok"}'
            "\n```"
        )
        output = agent._parse_response(content, "req-05")
        assert output.requirement_type == RequirementType.FUNCTIONAL
        assert output.priority == MoSCoWPriority.COULD_HAVE



class TestBaselineAgentRun:

    def test_run_preenche_prioritized_requirements(self, agent, sample_requirement):
        agent._llm.invoke = MagicMock(return_value=_mock_response(
            '{"requirement_type": "F", "nfr_category": null, "confidence": 0.9,'
            ' "classification_justification": "ok", "priority": "M",'
            ' "priority_score": 1.0, "priority_rank": 1, "priority_justification": "crítico"}'
        ))
        state = PipelineState(raw_requirements=[sample_requirement])
        result = agent.run(state)

        assert len(result.prioritized_requirements) == 1
        assert result.prioritized_requirements[0].priority == MoSCoWPriority.MUST_HAVE
        assert result.model_used == agent.model

    def test_run_aplica_ranking_global(self, agent):
        reqs = [
            Requirement(id=f"r{i}", text=f"Requisito de teste número {i} para o sistema.")
            for i in range(3)
        ]
        respostas = [
            '{"requirement_type": "F", "nfr_category": null, "confidence": 0.9, "classification_justification": "ok", "priority": "C", "priority_score": 0.5,  "priority_rank": 1, "priority_justification": "ok"}',
            '{"requirement_type": "F", "nfr_category": null, "confidence": 0.9, "classification_justification": "ok", "priority": "M", "priority_score": 1.0,  "priority_rank": 1, "priority_justification": "ok"}',
            '{"requirement_type": "F", "nfr_category": null, "confidence": 0.9, "classification_justification": "ok", "priority": "S", "priority_score": 0.75, "priority_rank": 1, "priority_justification": "ok"}',
        ]
        call_count = [0]

        def side_effect(_msgs):
            resp = MagicMock()
            resp.content = respostas[call_count[0] % len(respostas)]
            call_count[0] += 1
            return resp

        agent._llm.invoke = side_effect
        state = PipelineState(raw_requirements=reqs)
        result = agent.run(state)

        ranks = sorted(r.priority_rank for r in result.prioritized_requirements)
        assert ranks == list(range(1, len(reqs) + 1))

class TestBaselineAgentConfig:

    def test_sem_categorias_nfr(self):
        with patch("agents.baseline.build_llm"):
            agent = BaselineAgent(model="ollama/qwen2.5:7b", nfr_categories=None)
        assert agent._nfr_categories is None

    def test_com_categorias_injetadas(self):
        cats = [("SE", "Security"), ("PE", "Performance"), ("A", "Availability")]
        with patch("agents.baseline.build_llm"):
            agent = BaselineAgent(model="ollama/qwen2.5:7b", nfr_categories=cats)
        assert agent._nfr_categories == cats

    def test_lang_padrao_pt(self):
        with patch("agents.baseline.build_llm"):
            agent = BaselineAgent(model="ollama/qwen2.5:7b")
        assert agent._lang is Lang.PT

    def test_lang_en(self):
        with patch("agents.baseline.build_llm"):
            agent = BaselineAgent(model="ollama/qwen2.5:7b", lang=Lang.EN)
        assert agent._lang is Lang.EN