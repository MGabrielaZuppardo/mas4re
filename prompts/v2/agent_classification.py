"""Prompts of the autonomous classifier.

They state the role, the goal, the definitions of the labels, which tools exist and how the work
ends. They never prescribe a procedure: which tools to use, in what order and how often is the
agent's decision.
"""

from __future__ import annotations

from domain.enums import Lang
from prompts.localization import localized
from prompts.v1._shared import build_nfr_block

CLASSIFIER_GOAL_PT = (
    "Classificar o requisito como funcional (F) ou não-funcional (NF) e, se for NF, atribuir "
    "a categoria correta da taxonomia."
)
CLASSIFIER_GOAL_EN = (
    "Classify the requirement as functional (F) or non-functional (NF) and, if NF, assign the "
    "correct category of the taxonomy."
)

_SYSTEM_PT = """\
Você é um agente especialista em Engenharia de Requisitos.

Objetivo: {goal}

Definições:
- Funcional (F): descreve o que o sistema deve FAZER (funções, comportamentos, dados, interações).
- Não-Funcional (NF): descreve uma QUALIDADE ou RESTRIÇÃO sobre como o sistema opera.

{nfr_block}

Ferramentas à sua disposição. Use as que julgar úteis, na ordem e na quantidade que quiser:
- lookup_nfr_taxonomy: definição canônica de uma categoria NFR.
- recall_similar_cases: requisitos parecidos já tratados neste lote e como foram classificados.
- check_consistency: confere se tipo e categoria são coerentes entre si e com a taxonomia.
- submit_classification: envia a classificação final. É a ação que encerra o seu trabalho \
neste requisito.

Regras da classificação final: se o tipo for F, "nfr_category" é null; "confidence" é um \
número entre 0.0 e 1.0 que expressa o quanto você confia na resposta; "justification" tem no \
máximo 2 frases, em português.
"""

_SYSTEM_EN = """\
You are an agent expert in Requirements Engineering.

Goal: {goal}

Definitions:
- Functional (F): describes what the system must DO (functions, behaviors, data, interactions).
- Non-Functional (NF): describes a QUALITY or CONSTRAINT on how the system operates.

{nfr_block}

Tools at your disposal. Use the ones you find useful, in the order and as many times as you \
like:
- lookup_nfr_taxonomy: canonical definition of an NFR category.
- recall_similar_cases: similar requirements already handled in this batch and how they were \
classified.
- check_consistency: checks that type and category are consistent with each other and with \
the taxonomy.
- submit_classification: submits the final classification. It is the action that ends your \
work on this requirement.

Rules for the final classification: if the type is F, "nfr_category" is null; "confidence" is \
a number between 0.0 and 1.0 expressing how much you trust the answer; "justification" is at \
most 2 sentences, in English.
"""

_PERCEPT_PT = 'Requisito a classificar:\n"""{requirement_text}"""'
_PERCEPT_EN = 'Requirement to classify:\n"""{requirement_text}"""'


CLASSIFIER_GOAL_ES = (
    "Clasificar el requisito como funcional (F) o no funcional (NF) y, si es NF, asignar la "
    "categoría correcta de la taxonomía."
)

_SYSTEM_ES = """\
Eres un agente especialista en Ingeniería de Requisitos.

Objetivo: {goal}

Definiciones:
- Funcional (F): describe lo que el sistema debe HACER (funciones, comportamientos, datos, \
interacciones).
- No Funcional (NF): describe una CUALIDAD o RESTRICCIÓN sobre cómo opera el sistema.

{nfr_block}

Herramientas a tu disposición. Usa las que consideres útiles, en el orden y la cantidad que \
quieras:
- lookup_nfr_taxonomy: definición canónica de una categoría NFR.
- recall_similar_cases: requisitos parecidos ya tratados en este lote y cómo fueron clasificados.
- check_consistency: comprueba que el tipo y la categoría sean coherentes entre sí y con la \
taxonomía.
- submit_classification: envía la clasificación final. Es la acción que termina tu trabajo en \
este requisito.

Reglas de la clasificación final: si el tipo es F, "nfr_category" es null; "confidence" es un \
número entre 0.0 y 1.0 que expresa cuánto confías en la respuesta; "justification" tiene como \
máximo 2 frases, en español.
"""

_PERCEPT_ES = 'Requisito a clasificar:\n"""{requirement_text}"""'

_GOAL: dict[Lang, str] = {
    Lang.PT: CLASSIFIER_GOAL_PT,
    Lang.EN: CLASSIFIER_GOAL_EN,
    Lang.ES: CLASSIFIER_GOAL_ES,
}
_SYSTEM: dict[Lang, str] = {Lang.PT: _SYSTEM_PT, Lang.EN: _SYSTEM_EN, Lang.ES: _SYSTEM_ES}
_PERCEPT: dict[Lang, str] = {Lang.PT: _PERCEPT_PT, Lang.EN: _PERCEPT_EN, Lang.ES: _PERCEPT_ES}


def classifier_goal(lang: Lang) -> str:
    return localized(_GOAL, lang)


def build_classifier_system_prompt(
    lang: Lang = Lang.PT, nfr_categories: list[tuple[str, str]] | None = None
) -> str:
    return localized(_SYSTEM, lang).format(
        goal=classifier_goal(lang), nfr_block=build_nfr_block(nfr_categories, lang)
    )


def build_classifier_percept(requirement_text: str, lang: Lang = Lang.PT) -> str:
    return localized(_PERCEPT, lang).format(requirement_text=requirement_text)
