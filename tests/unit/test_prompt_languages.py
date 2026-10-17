"""Every prompt exists in every language, and a missing one is an error, not a fallback."""

from __future__ import annotations

import re
from types import ModuleType

import pytest

from domain.enums import Lang
from prompts import localization
from prompts.v1 import _shared, baseline, classification, prioritization
from prompts.v2 import (
    agent_classification,
    agent_classification_reply,
    agent_prioritization,
    mediation,
)

_TEXT = "El usuario debe poder exportar informes"
_ANSWER = {"requirement_type": "F", "nfr_category": None}
_PRIORITY_ANSWER = {"priority": "M", "priority_score": 1.0}
_UNFILLED_PLACEHOLDER = re.compile(r"\{[a-z_]+(?::[^}]*)?\}")
_MODULES: list[ModuleType] = [
    _shared,
    baseline,
    classification,
    prioritization,
    agent_classification,
    agent_classification_reply,
    agent_prioritization,
    mediation,
]


def _rendered(lang: Lang) -> dict[str, str]:
    """Every text a builder can hand to a model, in one language."""
    messages = {
        "classification": classification.build_classification_messages(_TEXT, lang),
        "classification_critique": classification.build_classification_critique_messages(
            _TEXT, _ANSWER, lang
        ),
        "prioritization": prioritization.build_prioritization_messages(
            _TEXT, "NF", "SE", 0.5, lang
        ),
        "prioritization_critique": prioritization.build_prioritization_critique_messages(
            _TEXT, _PRIORITY_ANSWER, lang
        ),
        "baseline": baseline.build_baseline_messages(_TEXT, lang),
    }
    texts = {
        f"{name}:{message['role']}": message["content"]
        for name, pair in messages.items()
        for message in pair
    }
    texts.update(
        {
            "nfr_block": _shared.build_nfr_block(None, lang),
            "classifier_goal": agent_classification.classifier_goal(lang),
            "classifier_system": agent_classification.build_classifier_system_prompt(lang),
            "classifier_percept": agent_classification.build_classifier_percept(_TEXT, lang),
            "reply_goal": agent_classification_reply.reply_goal(lang),
            "reply_system": agent_classification_reply.build_reply_system_prompt(lang),
            "reply_percept": agent_classification_reply.build_reply_percept(
                _TEXT, "NF", "SE", 0.6, "porque sí", "parece funcional", lang
            ),
            "prioritizer_goal": agent_prioritization.prioritizer_goal(lang),
            "prioritizer_system": agent_prioritization.build_prioritizer_system_prompt(lang),
            "prioritizer_report": agent_prioritization.describe_classification(
                "NF", "SE", 0.6, "porque sí", lang
            ),
            "prioritizer_percept": agent_prioritization.build_prioritizer_percept(
                _TEXT, "informe", lang
            ),
            "prioritizer_system_peers": agent_prioritization.build_prioritizer_system_prompt(
                lang, peer_requests=True
            ),
            "reconsideration_goal": agent_prioritization.reconsideration_goal(lang),
            "reconsideration_system": (
                agent_prioritization.build_prioritizer_reconsideration_system_prompt(lang)
            ),
            "reconsideration_percept": agent_prioritization.build_reconsideration_percept(
                _TEXT, "informe", "decisión previa", "respuesta del par", lang
            ),
            "priority_reply_goal": agent_prioritization.priority_reply_goal(lang),
            "priority_reply_system": agent_prioritization.build_prioritizer_reply_system_prompt(
                lang
            ),
            "priority_reply_percept": agent_prioritization.build_priority_reply_percept(
                _TEXT, "informe", "decisión previa", "motivo", "evidencia", lang
            ),
            "describe_priority": agent_prioritization.describe_priority(
                "M", 1.0, "porque sí", lang
            ),
            "peer_answer_agree": agent_prioritization.describe_peer_answer("agree", "motivo", lang),
            "peer_answer_refuse": agent_prioritization.describe_peer_answer(
                "refuse", "motivo", lang
            ),
            "mediator_reason_conflict": mediation.mediator_reason("conflict", lang),
            "mediator_reason_inflation": mediation.mediator_reason("must_inflation", lang),
            "conflict_evidence": mediation.conflict_evidence("SE", "C", lang),
            "inflation_evidence": mediation.inflation_evidence(0.8, 0.6, 0.4, lang),
        }
    )
    return texts


@pytest.fixture(scope="module")
def rendered() -> dict[Lang, dict[str, str]]:
    return {lang: _rendered(lang) for lang in Lang}


@pytest.mark.parametrize("lang", list(Lang))
def test_nenhum_texto_sai_vazio_nem_com_marcador_sem_preencher(lang, rendered):
    for name, text in rendered[lang].items():
        assert text.strip(), f"{lang}/{name} vazio"
        assert not _UNFILLED_PLACEHOLDER.search(text), f"{lang}/{name}: {text[:200]!r}"


@pytest.mark.parametrize("lang", [Lang.PT, Lang.EN])
def test_espanhol_nunca_e_copia_de_outro_idioma(lang, rendered):
    same = [
        name
        for name, text in rendered[Lang.ES].items()
        if text == rendered[lang][name] and name not in {"nfr_block"} and "percept" not in name
    ]

    assert same == []


def test_prompts_em_espanhol_pedem_a_justificativa_em_espanhol(rendered):
    asking = [
        name
        for name, text in rendered[Lang.ES].items()
        if "justification" in text and "en español" in text
    ]

    assert {"classification:system", "prioritization:system", "baseline:system"} <= set(asking)
    assert "classifier_system" in asking and "prioritizer_system" in asking


def test_o_texto_do_requisito_chega_ao_prompt_nos_tres_idiomas(rendered):
    for lang in Lang:
        for name in ("classification:user", "baseline:user", "classifier_percept"):
            assert _TEXT in rendered[lang][name]


def test_taxonomia_padrao_em_espanhol_tem_as_mesmas_dez_categorias():
    es = _shared.default_nfr_categories(Lang.ES)
    pt = _shared.default_nfr_categories(Lang.PT)

    assert [code for code, _ in es] == [code for code, _ in pt]
    assert dict(es)["PE"] == "Performance — rendimiento"


def _incomplete_tables(value: object, path: str) -> list[str]:
    """Paths of the tables keyed by language that lack one, looking inside nested tables too."""
    if not isinstance(value, dict) or not value:
        return []
    if all(isinstance(key, Lang) for key in value):
        return [] if set(value) == set(Lang) else [path]
    return [
        missing
        for key, inner in value.items()
        for missing in _incomplete_tables(inner, f"{path}[{key!r}]")
    ]


def test_toda_tabela_de_prompts_cobre_todos_os_idiomas():
    incomplete = [
        missing
        for module in _MODULES
        for name, value in vars(module).items()
        for missing in _incomplete_tables(value, f"{module.__name__}.{name}")
    ]

    assert incomplete == []


def test_o_verificador_de_tabelas_enxerga_idioma_faltando_em_tabela_aninhada():
    nested = {"work": {Lang.PT: "a", Lang.EN: "b", Lang.ES: "c"}, "reply": {Lang.PT: "a"}}

    assert _incomplete_tables(nested, "tabela") == ["tabela['reply']"]


def test_idioma_sem_texto_e_erro_explicito_e_nao_cai_em_outro():
    with pytest.raises(ValueError, match="'es'"):
        localization.localized({Lang.PT: "só português", Lang.EN: "english only"}, Lang.ES)
