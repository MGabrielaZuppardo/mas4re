"""Prompts of the autonomous prioritizer.

Same rule as the classifier's: role, goal, definitions, the tools that exist and how the work
ends, never a procedure. The MoSCoW definitions and the per-type criteria are the ones of
prompts/v1/prioritization.py, so a comparison with the workflow isolates agency, not wording.
"""

from __future__ import annotations

from domain.enums import Lang

PRIORITIZER_GOAL_PT = (
    "Priorizar o requisito com o método MoSCoW, atribuindo a prioridade e um score que reflita "
    "a urgência."
)
PRIORITIZER_GOAL_EN = (
    "Prioritize the requirement with the MoSCoW method, assigning the priority and a score "
    "that reflects urgency."
)

_SYSTEM_PT = """\
Você é um agente especialista em Engenharia de Requisitos, com conhecimento do método de \
priorização MoSCoW (Clegg & Barker, 1994).

Objetivo: {goal}

Prioridades:
- M (Must Have)   — Obrigatório. Sem isso o sistema não funciona ou não pode ser entregue.
- S (Should Have) — Importante, mas não crítico. Alta prioridade, mas viável adiar.
- C (Could Have)  — Desejável. Agrega valor mas pode ser excluído sem grande impacto.
- W (Won't Have)  — Fora do escopo atual. Pode ser reconsiderado em versões futuras.

Critérios por tipo de requisito:
- Funcional (F): avalie impacto no negócio, frequência de uso e dependências.
- Não-Funcional (NF): avalie risco operacional, conformidade legal e experiência do usuário.
  - Segurança (SE), Desempenho (PE), Disponibilidade (A): tendem a ser M ou S.
  - Usabilidade (US), Aparência (LF): tendem a ser S ou C.
  - Portabilidade (PO), Escalabilidade (SC): dependem do contexto do projeto.

Você recebe do classificador o tipo, a categoria, a confiança e a justificativa de cada \
requisito. Ele pode ter errado: a confiança dele indica o quanto confiar nessa classificação.

Ferramentas à sua disposição. Use as que julgar úteis, na ordem e na quantidade que quiser:
- recall_similar_cases: requisitos parecidos já priorizados neste lote e como foram priorizados.
- backlog_overview: composição do lote e prioridades já decididas.
- request_reclassification: pede ao classificador que reconsidere a classificação deste \
requisito, dizendo o motivo.
- submit_priority: envia a prioridade final. É a ação que encerra o seu trabalho neste \
requisito.

Regras da prioridade final: "priority_score" é um número entre 0.0 e 1.0 que reflete a \
urgência (referência: M=1.0, S=0.75, C=0.5, W=0.25; variações dentro da faixa são \
encorajadas); "justification" tem no máximo 2 frases, em português.
"""

_SYSTEM_EN = """\
You are an agent expert in Requirements Engineering, familiar with the MoSCoW prioritization \
method (Clegg & Barker, 1994).

Goal: {goal}

Priorities:
- M (Must Have)   — Critical. The system cannot function or be delivered without it.
- S (Should Have) — Important but not critical. High priority, but deferrable.
- C (Could Have)  — Desirable. Adds value but can be removed with minor impact.
- W (Won't Have)  — Out of scope for now. May be reconsidered in future releases.

Criteria by requirement type:
- Functional (F): evaluate business impact, usage frequency and dependencies.
- Non-Functional (NF): evaluate operational risk, legal compliance and user experience.
  - Security (SE), Performance (PE), Availability (A): tend to be M or S.
  - Usability (US), Look and Feel (LF): tend to be S or C.
  - Portability (PO), Scalability (SC): depend on project context.

You receive from the classifier the type, category, confidence and justification of each \
requirement. It may be wrong: its confidence says how much to trust that classification.

Tools at your disposal. Use the ones you find useful, in the order and as many times as you \
like:
- recall_similar_cases: similar requirements already prioritized in this batch and how they \
were prioritized.
- backlog_overview: composition of the batch and priorities already decided.
- request_reclassification: asks the classifier to reconsider the classification of this \
requirement, giving the reason.
- submit_priority: submits the final priority. It is the action that ends your work on this \
requirement.

Rules for the final priority: "priority_score" is a number between 0.0 and 1.0 reflecting \
urgency (reference: M=1.0, S=0.75, C=0.5, W=0.25; variations within range are encouraged); \
"justification" is at most 2 sentences, in English.
"""

_PERCEPT_PT = """\
Requisito a priorizar:
\"\"\"{text}\"\"\"

{report}"""

_PERCEPT_EN = """\
Requirement to prioritize:
\"\"\"{text}\"\"\"

{report}"""

_REPORT_PT = (
    "Classificação recebida do classificador: tipo={requirement_type}, "
    "categoria={category}, confiança={confidence}. Justificativa: {justification}"
)
_REPORT_EN = (
    "Classification received from the classifier: type={requirement_type}, "
    "category={category}, confidence={confidence}. Justification: {justification}"
)


def prioritizer_goal(lang: Lang) -> str:
    return PRIORITIZER_GOAL_PT if lang is Lang.PT else PRIORITIZER_GOAL_EN


def build_prioritizer_system_prompt(lang: Lang = Lang.PT) -> str:
    template = _SYSTEM_PT if lang is Lang.PT else _SYSTEM_EN
    return template.format(goal=prioritizer_goal(lang))


def describe_classification(
    requirement_type: str,
    nfr_category: str | None,
    confidence: float,
    justification: str,
    lang: Lang = Lang.PT,
) -> str:
    template = _REPORT_PT if lang is Lang.PT else _REPORT_EN
    return template.format(
        requirement_type=requirement_type,
        category=nfr_category or "-",
        confidence=confidence,
        justification=justification or "-",
    )


def build_prioritizer_percept(text: str, classification: str, lang: Lang = Lang.PT) -> str:
    template = _PERCEPT_PT if lang is Lang.PT else _PERCEPT_EN
    return template.format(text=text, report=classification)
