import pytest
from unittest.mock import MagicMock, patch
from domain.enums import RequirementType, NFRCategory
from domain.models import Requirement, ClassificationOutput
from agents.classifier import ClassificationAgent


@pytest.fixture
def agent():
    with patch("agents.classifier.build_llm"):
        return ClassificationAgent(model="ollama/qwen2.5:7b")


class TestClassificationAgentUnit:

    def test_parse_resposta_funcional(self, agent):
        content = '{"requirement_type": "F", "nfr_category": null, "confidence": 0.95, "justification": "Descreve comportamento funcional."}'
        output = agent._parse_response(content, "req-01")
        assert output.requirement_type == RequirementType.FUNCTIONAL
        assert output.nfr_category is None
        assert output.confidence == 0.95

    def test_parse_resposta_nfr(self, agent):
        content = '{"requirement_type": "NF", "nfr_category": "SE", "confidence": 0.88, "justification": "Requisito de segurança."}'
        output = agent._parse_response(content, "req-02")
        assert output.requirement_type == RequirementType.NON_FUNCTIONAL
        assert output.nfr_category == NFRCategory.SECURITY
        assert output.confidence == 0.88

    def test_parse_json_invalido_retorna_fallback(self, agent):
        output = agent._parse_response("resposta inválida", "req-03")
        assert output.requirement_type == RequirementType.FUNCTIONAL
        assert output.confidence == 0.0

    def test_parse_com_markdown(self, agent):
        content = '```json\n{"requirement_type": "F", "nfr_category": null, "confidence": 0.9, "justification": "ok"}\n```'
        output = agent._parse_response(content, "req-04")
        assert output.requirement_type == RequirementType.FUNCTIONAL