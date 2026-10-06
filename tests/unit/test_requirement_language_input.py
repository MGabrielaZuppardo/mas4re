"""Em condições EN, todo agente deve enviar ao LLM o texto original em inglês.

Regressão: o classificador e o baseline perderam o uso de text_en num merge,
enquanto o priorizador e o two-call o mantiveram, de modo que as condições EN
deixaram de ser comparáveis entre arquiteturas.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.baseline import BaselineAgent
from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from agents.two_call_baseline import TwoCallBaselineAgent
from domain.enums import Lang, RequirementType
from domain.models import ClassifiedRequirement, Requirement

TEXT_PT = "O sistema deve autenticar usuários."
TEXT_EN = "The system shall authenticate users."

BUILDERS = {
    "classifier": ["agents.classifier.build_classification_messages"],
    "baseline": ["agents.baseline.build_baseline_messages"],
    "prioritizer": ["agents.prioritizer.build_prioritization_messages"],
    "two_call": [
        "agents.two_call_baseline.build_classification_messages",
        "agents.two_call_baseline.build_prioritization_messages",
    ],
}


def _build(name: str, lang: Lang):
    modules = {
        "classifier": "agents.classifier",
        "baseline": "agents.baseline",
        "prioritizer": "agents.prioritizer",
        "two_call": "agents.two_call_baseline",
    }
    with patch(f"{modules[name]}.build_llm"):
        if name == "classifier":
            return ClassificationAgent(model="ollama/qwen2.5:7b", lang=lang)
        if name == "baseline":
            return BaselineAgent(model="ollama/qwen2.5:7b", lang=lang)
        if name == "prioritizer":
            return PrioritizationAgent(model="ollama/qwen2.5:7b", lang=lang)
        return TwoCallBaselineAgent(model="ollama/qwen2.5:7b", lang=lang)


def _item(name: str):
    requirement = Requirement(id="req-01", text=TEXT_PT, text_en=TEXT_EN)
    if name == "prioritizer":
        return ClassifiedRequirement.from_requirement(
            requirement,
            MagicMock(
                requirement_type=RequirementType.FUNCTIONAL,
                nfr_category=None,
                confidence=0.9,
                justification="",
            ),
        )
    return requirement


def _texts_sent_to_llm(name: str, lang: Lang) -> list[str]:
    agent = _build(name, lang)
    agent._llm = MagicMock()
    agent._llm.invoke.return_value = MagicMock(content="{}")
    sent: list[str] = []

    def capture(*args, **kwargs):
        sent.append(kwargs["requirement_text"])
        return []

    patches = [patch(target, side_effect=capture) for target in BUILDERS[name]]
    for p in patches:
        p.start()
    try:
        agent._process_single.__wrapped__(agent, _item(name))
    finally:
        for p in patches:
            p.stop()
    return sent


@pytest.mark.parametrize("name", list(BUILDERS))
def test_en_condition_sends_original_english_text(name):
    sent = _texts_sent_to_llm(name, Lang.EN)
    assert sent and all(text == TEXT_EN for text in sent)


@pytest.mark.parametrize("name", list(BUILDERS))
def test_pt_condition_sends_translated_text(name):
    sent = _texts_sent_to_llm(name, Lang.PT)
    assert sent and all(text == TEXT_PT for text in sent)
