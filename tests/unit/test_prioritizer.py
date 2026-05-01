import pytest
from unittest.mock import MagicMock, patch
from domain.enums import RequirementType, NFRCategory, MoSCoWPriority
from domain.models import (
    Requirement,
    ClassifiedRequirement,
    ClassificationOutput,
    PrioritizationOutput,
    PrioritizedRequirement,
    PipelineState,
)
from agents.prioritizer import PrioritizationAgent


@pytest.fixture
def agent():
    with patch("agents.prioritizer.build_llm"):
        return PrioritizationAgent(model="ollama/llama3.1:8b")


@pytest.fixture
def classified_requirements():
    reqs = [
        Requirement(id="req-01", text="O sistema deve autenticar usuários"),
        Requirement(id="req-02", text="O sistema deve responder em 200ms"),
        Requirement(id="req-03", text="O sistema deve gerar relatórios"),
    ]
    return [
        ClassifiedRequirement.from_requirement(
            req,
            ClassificationOutput(
                requirement_id=req.id,
                requirement_type=RequirementType.FUNCTIONAL,
                confidence=0.9,
                justification="ok",
            ),
        )
        for req in reqs
    ]


def make_prioritized(req, score: float, priority: MoSCoWPriority = MoSCoWPriority.MUST_HAVE):
    return PrioritizedRequirement.from_classified(
        req,
        PrioritizationOutput(
            requirement_id=req.id,
            priority=priority,
            priority_score=score,
            priority_rank=1,
            justification="ok",
        ),
    )


class TestPrioritizationAgentUnit:

    # ── _parse_response ───────────────────────────────────────────────────────

    def test_parse_must_have(self, agent):
        content = '{"priority": "M", "priority_score": 0.95, "priority_rank": 1, "justification": "Crítico."}'
        output = agent._parse_response(content, "req-01")
        assert output.priority == MoSCoWPriority.MUST_HAVE
        assert output.priority_score == 0.95

    def test_parse_should_have(self, agent):
        content = '{"priority": "S", "priority_score": 0.75, "priority_rank": 2, "justification": "Importante."}'
        output = agent._parse_response(content, "req-02")
        assert output.priority == MoSCoWPriority.SHOULD_HAVE

    def test_parse_could_have(self, agent):
        content = '{"priority": "C", "priority_score": 0.5, "priority_rank": 3, "justification": "Desejável."}'
        output = agent._parse_response(content, "req-03")
        assert output.priority == MoSCoWPriority.COULD_HAVE

    def test_parse_wont_have(self, agent):
        content = '{"priority": "W", "priority_score": 0.1, "priority_rank": 4, "justification": "Futuro."}'
        output = agent._parse_response(content, "req-04")
        assert output.priority == MoSCoWPriority.WONT_HAVE

    def test_parse_json_invalido_retorna_fallback(self, agent):
        output = agent._parse_response("resposta malformada", "req-05")
        assert output.priority == MoSCoWPriority.COULD_HAVE
        assert output.priority_score == 0.5

    def test_parse_com_markdown(self, agent):
        content = '```json\n{"priority": "M", "priority_score": 0.9, "priority_rank": 1, "justification": "ok"}\n```'
        output = agent._parse_response(content, "req-06")
        assert output.priority == MoSCoWPriority.MUST_HAVE

    def test_parse_justification_preenchida(self, agent):
        content = '{"priority": "S", "priority_score": 0.8, "priority_rank": 1, "justification": "Muito relevante."}'
        output = agent._parse_response(content, "req-07")
        assert output.justification == "Muito relevante."

    # ── run ───────────────────────────────────────────────────────────────────

    def test_run_atualiza_estado(self, agent, classified_requirements):
        state = PipelineState(
            run_id="run-test",
            raw_requirements=[],
            classified_requirements=classified_requirements,
        )
        with patch.object(agent, "prioritize_batch", return_value=[]) as mock:
            result = agent.run(state)
            mock.assert_called_once_with(classified_requirements, max_workers=3)

    def test_run_retorna_priorizados(self, agent, classified_requirements):
        mock_prioritized = [make_prioritized(classified_requirements[0], score=0.9)]
        state = PipelineState(
            run_id="run-test",
            raw_requirements=[],
            classified_requirements=classified_requirements,
        )
        with patch.object(agent, "prioritize_batch", return_value=mock_prioritized):
            result = agent.run(state)
            assert len(result.prioritized_requirements) == 1

    # ── prioritize_batch (ranking) ────────────────────────────────────────────

    def test_ranking_global_ordenado(self, agent, classified_requirements):
        scores = [0.5, 0.9, 0.7]
        def process_side_effect(req):
            idx = ["req-01", "req-02", "req-03"].index(req.id)
            return make_prioritized(req, score=scores[idx])

        with patch.object(agent, "_process_single", side_effect=process_side_effect):
            results = agent.prioritize_batch(classified_requirements)
            ranks = [r.priority_rank for r in results]
            scores_result = [r.priority_score for r in results]
            assert scores_result == sorted(scores_result, reverse=True)
            assert ranks == list(range(1, len(results) + 1))

    def test_prioritize_batch_ignora_falhas(self, agent, classified_requirements):
        def process_side_effect(req):
            if req.id == "req-01":
                raise RuntimeError("LLM indisponível")
            return make_prioritized(req, score=0.8)

        with patch.object(agent, "_process_single", side_effect=process_side_effect):
            results = agent.prioritize_batch(classified_requirements)
            ids = [r.id for r in results]
            assert "req-01" not in ids

    def test_prioritize_batch_todos_falham(self, agent, classified_requirements):
        with patch.object(agent, "_process_single", side_effect=RuntimeError("erro")):
            results = agent.prioritize_batch(classified_requirements)
            assert results == []

    # ── _process_single ───────────────────────────────────────────────────────

    def test_process_single_chama_llm(self, agent, classified_requirements):
        req = classified_requirements[0]
        mock_response = MagicMock()
        mock_response.content = '{"priority": "M", "priority_score": 0.95, "priority_rank": 1, "justification": "ok"}'
        agent._llm.invoke = MagicMock(return_value=mock_response)

        result = agent._process_single(req)
        agent._llm.invoke.assert_called_once()
        assert result.id == req.id

    def test_process_single_falha_propaga(self, agent, classified_requirements):
        req = classified_requirements[0]
        agent._llm.invoke = MagicMock(side_effect=ConnectionError("sem conexão"))

        with pytest.raises(ConnectionError):
            agent._process_single.__wrapped__(agent, req)