"""Testes unitários do PrioritizationAgent — sem chamadas LLM reais."""
import pytest
from unittest.mock import patch

from domain.enums import MoSCoWPriority
from domain.models import PrioritizationOutput
from agents.prioritizer import PrioritizationAgent


@pytest.fixture
def agent():
    with patch("agents.prioritizer.build_llm"):
        return PrioritizationAgent(model="ollama/llama3.1:8b")


class TestPrioritizationAgentUnit:

    def test_parse_must_have(self, agent):
        content = '{"priority": "M", "priority_score": 0.95, "priority_rank": 1, "justification": "Crítico para operação."}'
        output = agent._parse_response(content, "req-01")
        assert output.priority == MoSCoWPriority.MUST_HAVE
        assert output.priority_score == 0.95
        assert output.priority_rank == 1

    def test_parse_should_have(self, agent):
        content = '{"priority": "S", "priority_score": 0.75, "priority_rank": 1, "justification": "Importante mas não bloqueante."}'
        output = agent._parse_response(content, "req-02")
        assert output.priority == MoSCoWPriority.SHOULD_HAVE
        assert output.priority_score == 0.75

    def test_parse_could_have(self, agent):
        content = '{"priority": "C", "priority_score": 0.50, "priority_rank": 1, "justification": "Desejável."}'
        output = agent._parse_response(content, "req-03")
        assert output.priority == MoSCoWPriority.COULD_HAVE

    def test_parse_wont_have(self, agent):
        content = '{"priority": "W", "priority_score": 0.10, "priority_rank": 1, "justification": "Fora do escopo atual."}'
        output = agent._parse_response(content, "req-04")
        assert output.priority == MoSCoWPriority.WONT_HAVE
        assert output.priority_score == 0.10

    def test_parse_json_invalido_retorna_fallback(self, agent):
        output = agent._parse_response("resposta malformada", "req-05")
        assert output.priority == MoSCoWPriority.COULD_HAVE
        assert output.priority_score == 0.5
        assert "Parse falhou" in output.justification

    def test_parse_com_markdown(self, agent):
        content = '```json\n{"priority": "M", "priority_score": 1.0, "priority_rank": 1, "justification": "Essencial."}\n```'
        output = agent._parse_response(content, "req-06")
        assert output.priority == MoSCoWPriority.MUST_HAVE

    def test_parse_justification_preenchida(self, agent):
        content = '{"priority": "S", "priority_score": 0.8, "priority_rank": 1, "justification": "Relevante para UX."}'
        output = agent._parse_response(content, "req-07")
        assert len(output.justification) > 0

    def test_ranking_global_ordenado(self, agent):
        """Verifica que prioritize_batch reordena por priority_score decrescente."""
        from unittest.mock import MagicMock
        from domain.models import ClassifiedRequirement
        from domain.enums import RequirementType

        reqs = [
            ClassifiedRequirement(id=f"r{i}", text=f"Requisito {i}", requirement_type=RequirementType.FUNCTIONAL)
            for i in range(3)
        ]
        scores = [0.5, 0.9, 0.3]

        def fake_process(req):
            from domain.models import PrioritizedRequirement, PrioritizationOutput
            idx = int(req.id[1])
            out = PrioritizationOutput(
                requirement_id=req.id,
                priority=MoSCoWPriority.SHOULD_HAVE,
                priority_score=scores[idx],
                priority_rank=1,
            )
            return PrioritizedRequirement.from_classified(req, out)

        with patch.object(agent, "_process_single", side_effect=fake_process):
            result = agent.prioritize_batch(reqs, max_workers=1)

        assert result[0].priority_rank == 1
        assert result[0].priority_score == 0.9
        assert result[-1].priority_score == 0.3