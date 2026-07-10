from unittest.mock import MagicMock, patch

import pytest

from agents.two_call_baseline import TwoCallBaselineAgent
from domain.enums import Lang, MoSCoWPriority, RequirementType
from domain.models import PipelineState, Requirement


@pytest.fixture
def agent():
    with patch("agents.two_call_baseline.build_llm"):
        return TwoCallBaselineAgent(model="ollama/qwen2.5:7b")


@pytest.fixture
def sample_requirement():
    return Requirement(id="req-01", text="O sistema deve autenticar usuários via OAuth2.")


def _mock_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.content = content
    return mock


class TestParseClassification:
    def test_parse_valid_json(self, agent):
        content = (
            '{"requirement_type": "NF", "nfr_category": "SE", "confidence": 0.83,'
            ' "justification": "Requisito de segurança."}'
        )
        result = agent._parse_classification(content, "req-01")
        assert result.requirement_type == RequirementType.NON_FUNCTIONAL
        assert result.nfr_category == "SE"
        assert result.confidence == 0.83
        assert result.justification == "Requisito de segurança."

    def test_parse_invalid_json_returns_fallback(self, agent):
        result = agent._parse_classification("resposta inválida", "req-02")
        assert result.requirement_type == RequirementType.FUNCTIONAL
        assert result.confidence == 0.0
        assert "Parse falhou" in result.justification

    def test_parse_nfr_only_code_in_wrong_field(self, agent):
        """Schema fix: models sometimes put the NFR code (e.g. PE) directly in
        requirement_type instead of NF + nfr_category. Mirrors the same fix
        applied in ClassificationAgent (see agents/classifier.py)."""
        content = '{"requirement_type": "PE", "confidence": 0.85, "justification": "perf req"}'
        result = agent._parse_classification(content, "req-03")
        assert result.requirement_type == RequirementType.NON_FUNCTIONAL
        assert result.nfr_category == "PE"


class TestParsePrioritization:
    def test_parse_valid_json(self, agent):
        content = (
            '{"priority": "M", "priority_score": 0.95, "priority_rank": 1,'
            ' "justification": "Fluxo crítico."}'
        )
        result = agent._parse_prioritization(content, "req-01")
        assert result.priority == MoSCoWPriority.MUST_HAVE
        assert result.priority_score == 0.95
        assert result.justification == "Fluxo crítico."

    def test_parse_invalid_json_returns_fallback(self, agent):
        result = agent._parse_prioritization("resposta inválida", "req-02")
        assert result.priority == MoSCoWPriority.COULD_HAVE
        assert result.priority_score == 0.5
        assert "Parse falhou" in result.justification


class TestProcessSingle:
    def test_makes_two_sequential_calls(self, agent, sample_requirement):
        responses = [
            _mock_response(
                '{"requirement_type": "F", "nfr_category": null, "confidence": 0.4,'
                ' "justification": "ok"}'
            ),
            _mock_response(
                '{"priority": "S", "priority_score": 0.75, "priority_rank": 1,'
                ' "justification": "ok"}'
            ),
        ]
        agent._llm.invoke = MagicMock(side_effect=responses)

        output = agent._process_single(sample_requirement)

        assert agent._llm.invoke.call_count == 2
        assert output.requirement_type == RequirementType.FUNCTIONAL
        assert output.priority == MoSCoWPriority.SHOULD_HAVE
        # Original classifier confidence must reach the final output untouched —
        # only the *prioritization prompt* gets the routing disabled (below).
        assert output.confidence == 0.4

    def test_confidence_routing_is_disabled(self, agent, sample_requirement):
        """Core ablation property: unlike PrioritizationAgent in the real
        pipeline, this agent must always forward confidence=1.0 to the
        prioritization prompt, regardless of the classifier's actual
        confidence — see module docstring and agents/prioritizer.py."""
        agent._llm.invoke = MagicMock(
            side_effect=[
                _mock_response(
                    '{"requirement_type": "F", "nfr_category": null, "confidence": 0.1,'
                    ' "justification": "very unsure"}'
                ),
                _mock_response(
                    '{"priority": "C", "priority_score": 0.5, "priority_rank": 1,'
                    ' "justification": "ok"}'
                ),
            ]
        )

        with patch(
            "agents.two_call_baseline.build_prioritization_messages",
            wraps=None,
        ) as mock_build:
            mock_build.return_value = []
            agent._process_single(sample_requirement)

        assert mock_build.call_args.kwargs["confidence"] == 1.0


class TestRun:
    def test_run_preenche_prioritized_requirements(self, agent, sample_requirement):
        agent._llm.invoke = MagicMock(
            side_effect=[
                _mock_response(
                    '{"requirement_type": "F", "nfr_category": null, "confidence": 0.9,'
                    ' "justification": "ok"}'
                ),
                _mock_response(
                    '{"priority": "M", "priority_score": 1.0, "priority_rank": 1,'
                    ' "justification": "crítico"}'
                ),
            ]
        )
        state = PipelineState(raw_requirements=[sample_requirement])
        result = agent.run(state)

        assert len(result.prioritized_requirements) == 1
        assert result.prioritized_requirements[0].priority == MoSCoWPriority.MUST_HAVE
        assert result.model_used == agent.model


class TestConfig:
    def test_lang_padrao_pt(self):
        with patch("agents.two_call_baseline.build_llm"):
            agent = TwoCallBaselineAgent(model="ollama/qwen2.5:7b")
        assert agent._lang is Lang.PT

    def test_lang_en(self):
        with patch("agents.two_call_baseline.build_llm"):
            agent = TwoCallBaselineAgent(model="ollama/qwen2.5:7b", lang=Lang.EN)
        assert agent._lang is Lang.EN
