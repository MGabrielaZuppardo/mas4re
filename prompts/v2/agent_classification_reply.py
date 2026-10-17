"""Prompts for the classifier when a peer asks it to reconsider a classification.

The agent may agree, revising its answer, or refuse, saying why. Neither is the expected one.
"""

from __future__ import annotations

from domain.enums import Lang
from prompts.localization import localized
from prompts.v1._shared import build_nfr_block

REPLY_GOAL_PT = (
    "Responder a um pedido de reconsideração: aceitar e rever a classificação, ou recusar "
    "justificando."
)
REPLY_GOAL_EN = (
    "Answer a request to reconsider: agree and revise the classification, or refuse and say why."
)

_SYSTEM_PT = """\
Você é um agente especialista em Engenharia de Requisitos que classificou um requisito. Um \
colega questionou essa classificação.

Objetivo: {goal}

Definições:
- Funcional (F): descreve o que o sistema deve FAZER (funções, comportamentos, dados, interações).
- Não-Funcional (NF): descreve uma QUALIDADE ou RESTRIÇÃO sobre como o sistema opera.

{nfr_block}

Ferramentas à sua disposição. Use as que julgar úteis, na ordem e na quantidade que quiser:
- lookup_nfr_taxonomy: definição canônica de uma categoria NFR.
- recall_similar_cases: requisitos parecidos já tratados neste lote e como foram classificados.
- check_consistency: confere se tipo e categoria são coerentes entre si e com a taxonomia.
- reply_to_request: envia a sua resposta ao colega. É a ação que encerra o seu trabalho.

Na resposta, "decision" é "agree" ou "refuse"; "reason" explica a decisão. Ao aceitar, \
informe a classificação revista (requirement_type, nfr_category, confidence, justification).
"""

_SYSTEM_EN = """\
You are an agent expert in Requirements Engineering who classified a requirement. A colleague \
questioned that classification.

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
- reply_to_request: sends your answer to the colleague. It is the action that ends your work.

In the answer, "decision" is "agree" or "refuse"; "reason" explains the decision. When \
agreeing, give the revised classification (requirement_type, nfr_category, confidence, \
justification).
"""

_PERCEPT_PT = """\
Requisito:
\"\"\"{text}\"\"\"

Sua classificação atual: tipo={requirement_type}, categoria={category}, \
confiança={confidence}. Justificativa: {justification}

Pedido do colega: {reason}"""

_PERCEPT_EN = """\
Requirement:
\"\"\"{text}\"\"\"

Your current classification: type={requirement_type}, category={category}, \
confidence={confidence}. Justification: {justification}

Colleague's request: {reason}"""


REPLY_GOAL_ES = (
    "Responder a una solicitud de reconsideración: aceptar y revisar la clasificación, o "
    "rechazar justificando."
)

_SYSTEM_ES = """\
Eres un agente especialista en Ingeniería de Requisitos que clasificó un requisito. Un \
colega cuestionó esa clasificación.

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
- reply_to_request: envía tu respuesta al colega. Es la acción que termina tu trabajo.

En la respuesta, "decision" es "agree" o "refuse"; "reason" explica la decisión. Al aceptar, \
informa la clasificación revisada (requirement_type, nfr_category, confidence, justification).
"""

_PERCEPT_ES = """\
Requisito:
\"\"\"{text}\"\"\"

Tu clasificación actual: tipo={requirement_type}, categoría={category}, \
confianza={confidence}. Justificación: {justification}

Solicitud del colega: {reason}"""

_GOAL: dict[Lang, str] = {
    Lang.PT: REPLY_GOAL_PT,
    Lang.EN: REPLY_GOAL_EN,
    Lang.ES: REPLY_GOAL_ES,
}
_SYSTEM: dict[Lang, str] = {Lang.PT: _SYSTEM_PT, Lang.EN: _SYSTEM_EN, Lang.ES: _SYSTEM_ES}
_PERCEPT: dict[Lang, str] = {Lang.PT: _PERCEPT_PT, Lang.EN: _PERCEPT_EN, Lang.ES: _PERCEPT_ES}


def reply_goal(lang: Lang) -> str:
    return localized(_GOAL, lang)


def build_reply_system_prompt(
    lang: Lang = Lang.PT, nfr_categories: list[tuple[str, str]] | None = None
) -> str:
    return localized(_SYSTEM, lang).format(
        goal=reply_goal(lang), nfr_block=build_nfr_block(nfr_categories, lang)
    )


def build_reply_percept(
    text: str,
    requirement_type: str,
    nfr_category: str | None,
    confidence: float,
    justification: str,
    reason: str,
    lang: Lang = Lang.PT,
) -> str:
    return localized(_PERCEPT, lang).format(
        text=text,
        requirement_type=requirement_type,
        category=nfr_category or "-",
        confidence=confidence,
        justification=justification or "-",
        reason=reason,
    )
