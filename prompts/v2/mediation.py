"""What the mediator tells an agent when it asks for a decision to be reconsidered.

The mediator is part of the environment, not an agent: it points at a problem and gives the
evidence. Whether the evidence changes the decision is the agent's call.
"""

from __future__ import annotations

from domain.enums import Lang
from prompts.localization import localized

_REASON = {
    "conflict": {
        Lang.PT: "A categoria e a prioridade atribuídas a este requisito parecem conflitantes.",
        Lang.EN: "The category and the priority assigned to this requirement look conflicting.",
        Lang.ES: "La categoría y la prioridad asignadas a este requisito parecen contradictorias.",
    },
    "must_inflation": {
        Lang.PT: "Há Musts demais no backlog; reveja se este requisito é mesmo indispensável.",
        Lang.EN: "There are too many Musts in the backlog; check whether this requirement is "
        "really indispensable.",
        Lang.ES: "Hay demasiados Must en el backlog; revisa si este requisito es realmente "
        "indispensable.",
    },
}

_CONFLICT_EVIDENCE = {
    Lang.PT: "O classificador atribuiu a categoria crítica '{category}', mas a prioridade atual "
    "é '{priority}'. Requisitos dessa categoria costumam ser Must ou Should.",
    Lang.EN: "The classifier assigned the critical category '{category}', but the current "
    "priority is '{priority}'. Requirements of this category tend to be Must or Should.",
    Lang.ES: "El clasificador asignó la categoría crítica '{category}', pero la prioridad actual "
    "es '{priority}'. Los requisitos de esta categoría suelen ser Must o Should.",
}

_INFLATION_EVIDENCE = {
    Lang.PT: "{share:.0%} do backlog está como Must (referência: no máximo {limit:.0%}). Este "
    "requisito está entre os Musts de menor score ({score:.2f}).",
    Lang.EN: "{share:.0%} of the backlog is Must (reference: at most {limit:.0%}). This "
    "requirement is among the Musts with the lowest score ({score:.2f}).",
    Lang.ES: "El {share:.0%} del backlog está como Must (referencia: como máximo {limit:.0%}). "
    "Este requisito está entre los Must de menor score ({score:.2f}).",
}


def mediator_reason(kind: str, lang: Lang) -> str:
    return localized(_REASON[kind], lang)


def conflict_evidence(category: str, priority: str, lang: Lang) -> str:
    return localized(_CONFLICT_EVIDENCE, lang).format(category=category, priority=priority)


def inflation_evidence(share: float, limit: float, score: float, lang: Lang) -> str:
    return localized(_INFLATION_EVIDENCE, lang).format(share=share, limit=limit, score=score)
