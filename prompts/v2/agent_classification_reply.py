"""Prompts for the classifier when a peer asks it to reconsider a classification.

The agent may agree, revising its answer, or refuse, saying why. Neither is the expected one.
"""

from __future__ import annotations

from domain.enums import Lang
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


def reply_goal(lang: Lang) -> str:
    return REPLY_GOAL_PT if lang is Lang.PT else REPLY_GOAL_EN


def build_reply_system_prompt(
    lang: Lang = Lang.PT, nfr_categories: list[tuple[str, str]] | None = None
) -> str:
    template = _SYSTEM_PT if lang is Lang.PT else _SYSTEM_EN
    return template.format(goal=reply_goal(lang), nfr_block=build_nfr_block(nfr_categories, lang))


def build_reply_percept(
    text: str,
    requirement_type: str,
    nfr_category: str | None,
    confidence: float,
    justification: str,
    reason: str,
    lang: Lang = Lang.PT,
) -> str:
    template = _PERCEPT_PT if lang is Lang.PT else _PERCEPT_EN
    return template.format(
        text=text,
        requirement_type=requirement_type,
        category=nfr_category or "-",
        confidence=confidence,
        justification=justification or "-",
        reason=reason,
    )
