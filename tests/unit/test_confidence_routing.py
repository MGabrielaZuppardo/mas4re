"""Roteamento por confiança: o pipeline a repassa ao priorizador; o two-call a fixa em 1.0.

Documenta o comportamento do código que gerou os resultados do artigo: o limiar 0.70 só
dispara para confiança abaixo dele (nos resultados, 3 de 3750 predições do pipeline).
"""

from unittest.mock import MagicMock, patch

from agents.prioritizer import PrioritizationAgent
from agents.two_call_baseline import TwoCallBaselineAgent
from domain.enums import Lang, RequirementType
from domain.models import ClassifiedRequirement, Requirement
from prompts.v1.prioritization import build_prioritization_messages


def _classified(confidence: float) -> ClassifiedRequirement:
    return ClassifiedRequirement(
        text="O sistema deve autenticar usuários.",
        requirement_type=RequirementType.NON_FUNCTIONAL,
        nfr_category="SE",
        confidence=confidence,
    )


def _confidences_sent(builder_path: str, run) -> list[float]:
    sent: list[float] = []

    def capture(*args, **kwargs):
        sent.append(kwargs["confidence"])
        return []

    with patch(builder_path, side_effect=capture):
        run()
    return sent


def test_pipeline_prioritizer_passes_classifier_confidence():
    with patch("agents.prioritizer.build_llm"):
        agent = PrioritizationAgent(model="ollama/qwen2.5:7b", lang=Lang.PT)
    agent._llm = MagicMock()
    agent._llm.invoke.return_value = MagicMock(content="{}")

    sent = _confidences_sent(
        "agents.prioritizer.build_prioritization_messages",
        lambda: agent._process_single.__wrapped__(agent, _classified(0.55)),
    )

    assert sent == [0.55]


def test_two_call_baseline_fixes_confidence_at_one():
    with patch("agents.two_call_baseline.build_llm"):
        agent = TwoCallBaselineAgent(model="ollama/qwen2.5:7b", lang=Lang.PT)
    agent._llm = MagicMock()
    agent._llm.invoke.return_value = MagicMock(
        content='{"requirement_type": "NF", "nfr_category": "SE", "confidence": 0.30,'
        ' "justification": "x"}'
    )

    sent = _confidences_sent(
        "agents.two_call_baseline.build_prioritization_messages",
        lambda: agent._process_single.__wrapped__(agent, Requirement(text="Texto de teste aqui.")),
    )

    assert sent == [1.0]


def test_uncertainty_flag_appears_only_below_threshold():
    below = build_prioritization_messages("texto", "NF", "SE", confidence=0.69, lang=Lang.EN)
    at_threshold = build_prioritization_messages("texto", "NF", "SE", confidence=0.70, lang=Lang.EN)

    assert "UNCERTAIN" in str(below)
    assert "UNCERTAIN" not in str(at_threshold)
