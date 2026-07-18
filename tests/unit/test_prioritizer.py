from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from agents.prioritizer import PrioritizationAgent
from domain.enums import MoSCoWPriority, RequirementType
from domain.failures import FailureMode
from domain.models import (
    ClassificationOutput,
    ClassifiedRequirement,
    PipelineState,
    PrioritizationOutput,
    PrioritizedRequirement,
    Requirement,
)


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
        content = '{"priority": "M", "priority_score": 0.95, "priority_rank": 1, "justification": "Crítico."}'  # noqa: E501
        output = agent._parse_response(content, "req-01")
        assert output.priority == MoSCoWPriority.MUST_HAVE
        assert output.priority_score == 0.95

    def test_parse_should_have(self, agent):
        content = '{"priority": "S", "priority_score": 0.75, "priority_rank": 2, "justification": "Importante."}'  # noqa: E501
        output = agent._parse_response(content, "req-02")
        assert output.priority == MoSCoWPriority.SHOULD_HAVE

    def test_parse_could_have(self, agent):
        content = '{"priority": "C", "priority_score": 0.5, "priority_rank": 3, "justification": "Desejável."}'  # noqa: E501
        output = agent._parse_response(content, "req-03")
        assert output.priority == MoSCoWPriority.COULD_HAVE

    def test_parse_wont_have(self, agent):
        content = '{"priority": "W", "priority_score": 0.1, "priority_rank": 4, "justification": "Futuro."}'  # noqa: E501
        output = agent._parse_response(content, "req-04")
        assert output.priority == MoSCoWPriority.WONT_HAVE

    def test_parse_json_invalido_retorna_fallback(self, agent):
        output = agent._parse_response("resposta malformada", "req-05")
        assert output.priority == MoSCoWPriority.COULD_HAVE
        assert output.priority_score == 0.5

    def test_parse_com_markdown(self, agent):
        content = (
            '```json\n{"priority": "M", "priority_score": 0.9,'
            ' "priority_rank": 1, "justification": "ok"}\n```'
        )
        output = agent._parse_response(content, "req-06")
        assert output.priority == MoSCoWPriority.MUST_HAVE

    def test_parse_justification_preenchida(self, agent):
        content = (
            '{"priority": "S", "priority_score": 0.8, "priority_rank": 1,'
            ' "justification": "Muito relevante."}'
        )
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
            agent.run(state)
            mock.assert_called_once_with(classified_requirements)

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
        mock_response.content = (
            '{"priority": "M", "priority_score": 0.95, "priority_rank": 1, "justification": "ok"}'
        )
        agent._llm.invoke = MagicMock(return_value=mock_response)

        result = agent._process_single(req)
        agent._llm.invoke.assert_called_once()
        assert result.id == req.id

    def test_process_single_falha_propaga(self, agent, classified_requirements):
        req = classified_requirements[0]
        agent._llm.invoke = MagicMock(side_effect=ConnectionError("sem conexão"))

        with pytest.raises(ConnectionError):
            agent._process_single.__wrapped__(agent, req)

    def test_process_single_priority_score_invalido_propaga_validation_error(
        self, agent, classified_requirements
    ):
        """Structured-but-invalid output (priority_score out of [0,1]) must NOT
        be silently swallowed into the safe fallback like a JSON decode
        failure — it should propagate as a ValidationError so @llm_retry
        re-queries the LLM and, on exhaustion, DetectorChain sees
        parsed_ok=False."""
        req = classified_requirements[0]
        mock_response = MagicMock()
        mock_response.content = (
            '{"priority": "M", "priority_score": 1.5, "priority_rank": 1, "justification": "ok"}'
        )
        agent._llm.invoke = MagicMock(return_value=mock_response)

        with pytest.raises(ValidationError):
            agent._process_single.__wrapped__(agent, req)

    # ── failure detection (SQ3, ADR-003) ──────────────────────────────────────

    def test_prioritize_batch_falha_registra_schema_invalid(self, agent, classified_requirements):
        with patch.object(agent, "_process_single", side_effect=RuntimeError("erro")):
            agent.prioritize_batch(classified_requirements)

        modes = {r.mode for r in agent._failure_records}
        assert FailureMode.SCHEMA_INVALID in modes

    def test_run_persiste_failure_detections_no_estado(self, agent, classified_requirements):
        state = PipelineState(
            run_id="run-test",
            raw_requirements=[],
            classified_requirements=classified_requirements,
        )
        with patch.object(agent, "_process_single", side_effect=RuntimeError("erro")):
            result = agent.run(state)

        assert "failure_detections" in result.metrics
        assert len(result.metrics["failure_detections"]) == len(classified_requirements)
