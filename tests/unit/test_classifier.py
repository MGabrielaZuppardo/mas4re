from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from agents.classifier import ClassificationAgent
from domain.enums import NFRCategory, RequirementType
from domain.failures import FailureMode
from domain.models import ClassificationOutput, ClassifiedRequirement, PipelineState, Requirement


@pytest.fixture
def agent():
    with patch("agents.classifier.build_llm"):
        return ClassificationAgent(model="ollama/qwen2.5:7b")


@pytest.fixture
def requirements():
    return [
        Requirement(id="req-01", text="O sistema deve autenticar usuários"),
        Requirement(id="req-02", text="O sistema deve responder em 200ms"),
    ]


class TestClassificationAgentUnit:
    def test_parse_resposta_funcional(self, agent):
        content = (
            '{"requirement_type": "F", "nfr_category": null, "confidence": 0.95,'
            ' "justification": "Descreve comportamento funcional."}'
        )
        output = agent._parse_response(content, "req-01")
        assert output.requirement_type == RequirementType.FUNCTIONAL
        assert output.nfr_category is None
        assert output.confidence == 0.95

    def test_parse_resposta_nfr(self, agent):
        content = (
            '{"requirement_type": "NF", "nfr_category": "SE", "confidence": 0.88,'
            ' "justification": "Requisito de segurança."}'
        )
        output = agent._parse_response(content, "req-02")
        assert output.requirement_type == RequirementType.NON_FUNCTIONAL
        assert output.nfr_category == NFRCategory.SECURITY
        assert output.confidence == 0.88

    def test_parse_json_invalido_retorna_fallback(self, agent):
        output = agent._parse_response("resposta inválida", "req-03")
        assert output.requirement_type == RequirementType.FUNCTIONAL
        assert output.confidence == 0.0

    def test_parse_com_markdown(self, agent):
        content = (
            '```json\n{"requirement_type": "F", "nfr_category": null,'
            ' "confidence": 0.9, "justification": "ok"}\n```'
        )
        output = agent._parse_response(content, "req-04")
        assert output.requirement_type == RequirementType.FUNCTIONAL

    def test_run_atualiza_estado(self, agent, requirements):
        state = PipelineState(run_id="run-test", raw_requirements=requirements)
        with patch.object(agent, "classify_batch", return_value=[]) as mock:
            result = agent.run(state)
            mock.assert_called_once_with(requirements)
            assert result.model_used == agent.model

    def test_run_retorna_classificados(self, agent, requirements):
        mock_classified = [
            ClassifiedRequirement.from_requirement(
                requirements[0],
                ClassificationOutput(
                    requirement_id="req-01",
                    requirement_type=RequirementType.FUNCTIONAL,
                    confidence=0.9,
                    justification="teste",
                ),
            )
        ]
        state = PipelineState(run_id="run-test", raw_requirements=requirements)
        with patch.object(agent, "classify_batch", return_value=mock_classified):
            result = agent.run(state)
            assert len(result.classified_requirements) == 1

    def test_classify_batch_ignora_falhas(self, agent, requirements):
        def process_side_effect(req):
            if req.id == "req-01":
                raise RuntimeError("LLM indisponível")
            return ClassifiedRequirement.from_requirement(
                req,
                ClassificationOutput(
                    requirement_id=req.id,
                    requirement_type=RequirementType.FUNCTIONAL,
                    confidence=0.8,
                    justification="ok",
                ),
            )

        with patch.object(agent, "_process_single", side_effect=process_side_effect):
            results = agent.classify_batch(requirements)
            assert len(results) == 1
            assert results[0].id == "req-02"

    def test_classify_batch_todos_falham(self, agent, requirements):
        with patch.object(agent, "_process_single", side_effect=RuntimeError("erro")):
            results = agent.classify_batch(requirements)
            assert results == []

    def test_process_single_chama_llm(self, agent):
        req = Requirement(id="req-01", text="O sistema deve logar eventos")
        mock_response = MagicMock()
        mock_response.content = (
            '{"requirement_type": "F", "confidence": 0.9, "justification": "funcional"}'
        )
        agent._llm.invoke = MagicMock(return_value=mock_response)

        result = agent._process_single(req)
        agent._llm.invoke.assert_called_once()
        assert result.id == "req-01"

    def test_process_single_falha_propaga(self, agent):
        req = Requirement(id="req-01", text="texto")
        agent._llm.invoke = MagicMock(side_effect=ConnectionError("sem conexão"))

        with pytest.raises(ConnectionError):
            agent._process_single.__wrapped__(agent, req)

    def test_process_single_confidence_invalida_propaga_validation_error(self, agent):
        """Structured-but-invalid output (confidence out of [0,1]) must NOT be
        silently swallowed into the safe fallback like a JSON decode failure —
        it should propagate as a ValidationError so @llm_retry re-queries the
        LLM and, on exhaustion, DetectorChain sees parsed_ok=False."""
        req = Requirement(id="req-01", text="texto")
        mock_response = MagicMock()
        mock_response.content = (
            '{"requirement_type": "F", "confidence": 1.4, "justification": "ok"}'
        )
        agent._llm.invoke = MagicMock(return_value=mock_response)

        with pytest.raises(ValidationError):
            agent._process_single.__wrapped__(agent, req)

    # ── failure detection (SQ3, ADR-003) ──────────────────────────────────────

    def test_run_registra_baixa_confianca_em_failure_detections(self, agent, requirements):
        def process_side_effect(req):
            return ClassifiedRequirement.from_requirement(
                req,
                ClassificationOutput(
                    requirement_id=req.id,
                    requirement_type=RequirementType.FUNCTIONAL,
                    confidence=0.2,
                    justification="ok",
                ),
            )

        state = PipelineState(run_id="run-test", raw_requirements=requirements)
        with patch.object(agent, "_process_single", side_effect=process_side_effect):
            result = agent.run(state)

        detections = result.metrics["failure_detections"]
        modes = {d["mode"] for d in detections}
        assert FailureMode.LOW_CONFIDENCE.value in modes

    def test_classify_batch_falha_registra_schema_invalid(self, agent, requirements):
        with patch.object(agent, "_process_single", side_effect=RuntimeError("erro")):
            agent.classify_batch(requirements)

        modes = {r.mode for r in agent._failure_records}
        assert FailureMode.SCHEMA_INVALID in modes
