from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from agents.classifier import ClassificationAgent
from domain.enums import NFRCategory, RequirementType
from domain.failures import FailureMode
from domain.models import ClassificationOutput, ClassifiedRequirement, PipelineState, Requirement

_NO_REVISION = MagicMock()
_NO_REVISION.content = '{"needs_revision": false, "issue": "", "revised_output": null}'


def _primary_response(content: str) -> MagicMock:
    resp = MagicMock(tool_calls=[])
    resp.content = content
    return resp


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
        agent._llm_with_tools.invoke = MagicMock(
            return_value=_primary_response(
                '{"requirement_type": "F", "confidence": 0.9, "justification": "funcional"}'
            )
        )
        agent._llm.invoke = MagicMock(return_value=_NO_REVISION)

        result = agent._process_single(req)
        agent._llm_with_tools.invoke.assert_called_once()
        assert result.id == "req-01"

    def test_process_single_falha_propaga(self, agent):
        req = Requirement(id="req-01", text="texto")
        agent._llm_with_tools.invoke = MagicMock(side_effect=ConnectionError("sem conexão"))

        with pytest.raises(ConnectionError):
            agent._process_single.__wrapped__(agent, req)

    def test_process_single_confidence_invalida_propaga_validation_error(self, agent):
        """Structured-but-invalid output (confidence out of [0,1]) must NOT be
        silently swallowed into the safe fallback like a JSON decode failure —
        it should propagate as a ValidationError so @llm_retry re-queries the
        LLM and, on exhaustion, DetectorChain sees parsed_ok=False."""
        req = Requirement(id="req-01", text="texto")
        agent._llm_with_tools.invoke = MagicMock(
            return_value=_primary_response(
                '{"requirement_type": "F", "confidence": 1.4, "justification": "ok"}'
            )
        )

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


class TestClassificationAgentAgentic:
    """ADR-011: memory, tool-use (ReAct), self-critique (Self-Refine)."""

    def test_critique_sem_revisao_mantem_saida_original(self, agent):
        req = Requirement(id="req-01", text="texto")
        agent._llm_with_tools.invoke = MagicMock(
            return_value=_primary_response(
                '{"requirement_type": "F", "confidence": 0.9, "justification": "ok"}'
            )
        )
        agent._llm.invoke = MagicMock(return_value=_NO_REVISION)

        result = agent._process_single(req)

        assert result.requirement_type == RequirementType.FUNCTIONAL
        assert result.confidence == 0.9

    def test_critique_com_revisao_usa_saida_revisada(self, agent):
        req = Requirement(id="req-01", text="texto")
        agent._llm_with_tools.invoke = MagicMock(
            return_value=_primary_response(
                '{"requirement_type": "F", "confidence": 0.5, "justification": "duvidoso"}'
            )
        )
        revision = MagicMock()
        revision.content = (
            '{"needs_revision": true, "issue": "categoria errada", "revised_output": '
            '{"requirement_type": "NF", "nfr_category": "SE", "confidence": 0.95, '
            '"justification": "na verdade eh seguranca"}}'
        )
        agent._llm.invoke = MagicMock(return_value=revision)

        result = agent._process_single(req)

        assert result.requirement_type == RequirementType.NON_FUNCTIONAL
        assert result.nfr_category == NFRCategory.SECURITY
        assert result.confidence == 0.95

    def test_tool_chamada_usa_observacao_no_fechamento(self, agent):
        req = Requirement(id="req-01", text="O sistema deve criptografar dados em repouso")
        tool_call_response = MagicMock(
            tool_calls=[
                {"name": "lookup_nfr_taxonomy", "args": {"category_code": "SE"}, "id": "call-1"}
            ]
        )
        tool_call_response.content = ""
        agent._llm_with_tools.invoke = MagicMock(return_value=tool_call_response)

        final_response = _primary_response(
            '{"requirement_type": "NF", "nfr_category": "SE", "confidence": 0.95,'
            ' "justification": "confirmado via taxonomia"}'
        )
        agent._llm.invoke = MagicMock(side_effect=[final_response, _NO_REVISION])

        result = agent._process_single(req)

        assert agent._llm.invoke.call_count == 2
        assert result.nfr_category == NFRCategory.SECURITY
        followup_messages = agent._llm.invoke.call_args_list[0].args[0]
        tool_messages = [m for m in followup_messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert "SE" in tool_messages[0].content

    def test_memoria_injeta_few_shot_no_item_seguinte(self, agent):
        reqs = [
            Requirement(id="req-01", text="O sistema deve criptografar dados sensiveis"),
            Requirement(id="req-02", text="O sistema deve criptografar backups tambem"),
        ]
        agent._llm_with_tools.invoke = MagicMock(
            side_effect=lambda messages: _primary_response(
                '{"requirement_type": "NF", "nfr_category": "SE", "confidence": 0.9,'
                ' "justification": "seguranca"}'
            )
        )
        agent._llm.invoke = MagicMock(return_value=_NO_REVISION)

        agent.classify_batch(reqs)

        assert agent._llm_with_tools.invoke.call_count == 2
        second_call_messages = agent._llm_with_tools.invoke.call_args_list[1].args[0]
        user_content = next(m["content"] for m in second_call_messages if m["role"] == "user")
        assert "criptografar dados sensiveis" in user_content

    def test_batch_maior_que_warmup_congela_memoria_apos_warmup(self, agent):
        reqs = [
            Requirement(id=f"req-{i:02d}", text=f"O sistema deve fazer a coisa numero {i}")
            for i in range(12)
        ]
        agent._llm_with_tools.invoke = MagicMock(
            side_effect=lambda messages: _primary_response(
                '{"requirement_type": "F", "confidence": 0.9, "justification": "ok"}'
            )
        )
        agent._llm.invoke = MagicMock(return_value=_NO_REVISION)

        results = agent.classify_batch(reqs)

        assert len(results) == 12
        assert agent._memory.frozen is True
        assert len(agent._memory._items) == 10  # only warm-up items recorded
