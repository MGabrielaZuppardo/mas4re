"""Prompts of the autonomous prioritizer.

Same rule as the classifier's: role, goal, definitions, the tools that exist and how the work
ends, never a procedure. The MoSCoW definitions and the per-type criteria are the ones of
prompts/v1/prioritization.py, so a comparison with the workflow isolates agency, not wording.

The list of tools is composed from the actions the agent really has: the prompt never mentions a
tool the protocol of the run does not give it (a promised channel nobody answers is an invented
affordance, and the agent would waste steps on it).

Three kinds of work share one definition block: prioritizing an item, deciding again after a peer
answered the agent's request, and answering a request of the mediator.
"""

from __future__ import annotations

from domain.enums import Lang
from prompts.localization import localized

# ── Goals ──────────────────────────────────────────────────────────────────────

PRIORITIZER_GOAL_PT = (
    "Priorizar o requisito com o método MoSCoW, atribuindo a prioridade e um score que reflita "
    "a urgência."
)
PRIORITIZER_GOAL_EN = (
    "Prioritize the requirement with the MoSCoW method, assigning the priority and a score "
    "that reflects urgency."
)
PRIORITIZER_GOAL_ES = (
    "Priorizar el requisito con el método MoSCoW, asignando la prioridad y un score que "
    "refleje la urgencia."
)

RECONSIDERATION_GOAL_PT = (
    "Decidir de novo a prioridade do requisito, levando em conta a resposta do classificador ao "
    "seu pedido."
)
RECONSIDERATION_GOAL_EN = (
    "Decide the priority of the requirement again, taking into account the classifier's answer "
    "to your request."
)
RECONSIDERATION_GOAL_ES = (
    "Decidir de nuevo la prioridad del requisito, teniendo en cuenta la respuesta del "
    "clasificador a tu solicitud."
)

PRIORITY_REPLY_GOAL_PT = (
    "Responder a um pedido de reconsideração da prioridade: aceitar e revê-la, ou recusar "
    "justificando."
)
PRIORITY_REPLY_GOAL_EN = (
    "Answer a request to reconsider a priority: agree and revise it, or refuse and say why."
)
PRIORITY_REPLY_GOAL_ES = (
    "Responder a una solicitud de reconsideración de la prioridad: aceptar y revisarla, o "
    "rechazar justificando."
)

_GOAL = {
    "work": {
        Lang.PT: PRIORITIZER_GOAL_PT,
        Lang.EN: PRIORITIZER_GOAL_EN,
        Lang.ES: PRIORITIZER_GOAL_ES,
    },
    "reconsideration": {
        Lang.PT: RECONSIDERATION_GOAL_PT,
        Lang.EN: RECONSIDERATION_GOAL_EN,
        Lang.ES: RECONSIDERATION_GOAL_ES,
    },
    "reply": {
        Lang.PT: PRIORITY_REPLY_GOAL_PT,
        Lang.EN: PRIORITY_REPLY_GOAL_EN,
        Lang.ES: PRIORITY_REPLY_GOAL_ES,
    },
}

# ── System prompt: definitions shared by every kind of work ────────────────────

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
{tools}

{closing_rules}
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
{tools}

{closing_rules}
"""

_SYSTEM_ES = """\
Eres un agente especialista en Ingeniería de Requisitos, con conocimiento del método de \
priorización MoSCoW (Clegg & Barker, 1994).

Objetivo: {goal}

Prioridades:
- M (Must Have)   — Obligatorio. Sin esto el sistema no funciona o no puede entregarse.
- S (Should Have) — Importante, pero no crítico. Alta prioridad, aunque se puede posponer.
- C (Could Have)  — Deseable. Aporta valor, pero puede excluirse sin gran impacto.
- W (Won't Have)  — Fuera del alcance actual. Puede reconsiderarse en versiones futuras.

Criterios según el tipo de requisito:
- Funcional (F): evalúa el impacto en el negocio, la frecuencia de uso y las dependencias.
- No Funcional (NF): evalúa el riesgo operativo, el cumplimiento legal y la experiencia del \
usuario.
  - Seguridad (SE), Rendimiento (PE), Disponibilidad (A): tienden a ser M o S.
  - Usabilidad (US), Apariencia (LF): tienden a ser S o C.
  - Portabilidad (PO), Escalabilidad (SC): dependen del contexto del proyecto.

Recibes del clasificador el tipo, la categoría, la confianza y la justificación de cada \
requisito. Puede haberse equivocado: su confianza indica cuánto fiarse de esa clasificación.

Herramientas a tu disposición. Usa las que consideres útiles, en el orden y la cantidad que \
quieras:
{tools}

{closing_rules}
"""

_SYSTEM = {Lang.PT: _SYSTEM_PT, Lang.EN: _SYSTEM_EN, Lang.ES: _SYSTEM_ES}

# ── Tools, one line each ───────────────────────────────────────────────────────

_TOOL_LINES = {
    "recall_similar_cases": {
        Lang.PT: "- recall_similar_cases: requisitos parecidos já priorizados neste lote e como "
        "foram priorizados.",
        Lang.EN: "- recall_similar_cases: similar requirements already prioritized in this batch "
        "and how they were prioritized.",
        Lang.ES: "- recall_similar_cases: requisitos parecidos ya priorizados en este lote y cómo "
        "fueron priorizados.",
    },
    "backlog_overview": {
        Lang.PT: "- backlog_overview: composição do lote e prioridades já decididas.",
        Lang.EN: "- backlog_overview: composition of the batch and priorities already decided.",
        Lang.ES: "- backlog_overview: composición del lote y prioridades ya decididas.",
    },
    "request_reclassification": {
        Lang.PT: "- request_reclassification: pede ao classificador que reconsidere a "
        "classificação deste requisito, dizendo o motivo.",
        Lang.EN: "- request_reclassification: asks the classifier to reconsider the "
        "classification of this requirement, giving the reason.",
        Lang.ES: "- request_reclassification: pide al clasificador que reconsidere la "
        "clasificación de este requisito, indicando el motivo.",
    },
    "submit_priority": {
        Lang.PT: "- submit_priority: envia a prioridade final. É a ação que encerra o seu "
        "trabalho neste requisito.",
        Lang.EN: "- submit_priority: submits the final priority. It is the action that ends your "
        "work on this requirement.",
        Lang.ES: "- submit_priority: envía la prioridad final. Es la acción que termina tu "
        "trabajo en este requisito.",
    },
    "reply_to_request": {
        Lang.PT: "- reply_to_request: envia a sua resposta ao pedido. É a ação que encerra o seu "
        "trabalho neste requisito.",
        Lang.EN: "- reply_to_request: sends your answer to the request. It is the action that "
        "ends your work on this requirement.",
        Lang.ES: "- reply_to_request: envía tu respuesta a la solicitud. Es la acción que "
        "termina tu trabajo en este requisito.",
    },
}

# ── Closing rules ──────────────────────────────────────────────────────────────

_FINAL_PRIORITY_RULES = {
    Lang.PT: 'Regras da prioridade final: "priority_score" é um número entre 0.0 e 1.0 que '
    "reflete a urgência (referência: M=1.0, S=0.75, C=0.5, W=0.25; variações dentro da faixa "
    'são encorajadas); "justification" tem no máximo 2 frases, em português.',
    Lang.EN: 'Rules for the final priority: "priority_score" is a number between 0.0 and 1.0 '
    "reflecting urgency (reference: M=1.0, S=0.75, C=0.5, W=0.25; variations within range are "
    'encouraged); "justification" is at most 2 sentences, in English.',
    Lang.ES: 'Reglas de la prioridad final: "priority_score" es un número entre 0.0 y 1.0 que '
    "refleja la urgencia (referencia: M=1.0, S=0.75, C=0.5, W=0.25; se alientan variaciones "
    'dentro del rango); "justification" tiene como máximo 2 frases, en español.',
}

_REPLY_RULES = {
    Lang.PT: 'Na resposta, "decision" é "agree" ou "refuse"; "reason" explica a decisão. Ao '
    'aceitar, informe a prioridade revista ("priority", "priority_score", "justification"); ao '
    "recusar, a prioridade atual é mantida.",
    Lang.EN: 'In the answer, "decision" is "agree" or "refuse"; "reason" explains the decision. '
    'When agreeing, give the revised priority ("priority", "priority_score", "justification"); '
    "when refusing, the current priority stays.",
    Lang.ES: 'En la respuesta, "decision" es "agree" o "refuse"; "reason" explica la decisión. '
    'Al aceptar, informa la prioridad revisada ("priority", "priority_score", "justification"); '
    "al rechazar, se mantiene la prioridad actual.",
}

# ── Percepts ───────────────────────────────────────────────────────────────────

_PERCEPT_PT = """\
Requisito a priorizar:
\"\"\"{text}\"\"\"

{report}"""

_PERCEPT_EN = """\
Requirement to prioritize:
\"\"\"{text}\"\"\"

{report}"""

_PERCEPT_ES = """\
Requisito a priorizar:
\"\"\"{text}\"\"\"

{report}"""

_PERCEPT = {Lang.PT: _PERCEPT_PT, Lang.EN: _PERCEPT_EN, Lang.ES: _PERCEPT_ES}

_RECONSIDERATION_PERCEPT = {
    Lang.PT: _PERCEPT_PT + "\n\nSua prioridade anterior: {previous}\n\n{answer}",
    Lang.EN: _PERCEPT_EN + "\n\nYour previous priority: {previous}\n\n{answer}",
    Lang.ES: _PERCEPT_ES + "\n\nTu prioridad anterior: {previous}\n\n{answer}",
}

_REPLY_PERCEPT = {
    Lang.PT: """\
Requisito:
\"\"\"{text}\"\"\"

{report}

Sua prioridade atual: {previous}

Pedido recebido: {reason}
Evidência: {evidence}""",
    Lang.EN: """\
Requirement:
\"\"\"{text}\"\"\"

{report}

Your current priority: {previous}

Request received: {reason}
Evidence: {evidence}""",
    Lang.ES: """\
Requisito:
\"\"\"{text}\"\"\"

{report}

Tu prioridad actual: {previous}

Solicitud recibida: {reason}
Evidencia: {evidence}""",
}

_REPORT_PT = (
    "Classificação recebida do classificador: tipo={requirement_type}, "
    "categoria={category}, confiança={confidence}. Justificativa: {justification}"
)
_REPORT_EN = (
    "Classification received from the classifier: type={requirement_type}, "
    "category={category}, confidence={confidence}. Justification: {justification}"
)
_REPORT_ES = (
    "Clasificación recibida del clasificador: tipo={requirement_type}, "
    "categoría={category}, confianza={confidence}. Justificación: {justification}"
)
_REPORT = {Lang.PT: _REPORT_PT, Lang.EN: _REPORT_EN, Lang.ES: _REPORT_ES}

_PRIORITY = {
    Lang.PT: "prioridade={priority}, score={score}. Justificativa: {justification}",
    Lang.EN: "priority={priority}, score={score}. Justification: {justification}",
    Lang.ES: "prioridad={priority}, score={score}. Justificación: {justification}",
}

_PEER_ANSWER = {
    "agree": {
        Lang.PT: "O classificador aceitou o seu pedido e revisou a classificação (acima). "
        "Motivo: {reason}",
        Lang.EN: "The classifier agreed to your request and revised the classification (above). "
        "Reason: {reason}",
        Lang.ES: "El clasificador aceptó tu solicitud y revisó la clasificación (arriba). "
        "Motivo: {reason}",
    },
    "refuse": {
        Lang.PT: "O classificador recusou o seu pedido e manteve a classificação. Motivo: {reason}",
        Lang.EN: "The classifier refused your request and kept the classification. "
        "Reason: {reason}",
        Lang.ES: "El clasificador rechazó tu solicitud y mantuvo la clasificación. "
        "Motivo: {reason}",
    },
}


# ── Builders ───────────────────────────────────────────────────────────────────


def prioritizer_goal(lang: Lang) -> str:
    return localized(_GOAL["work"], lang)


def reconsideration_goal(lang: Lang) -> str:
    return localized(_GOAL["reconsideration"], lang)


def priority_reply_goal(lang: Lang) -> str:
    return localized(_GOAL["reply"], lang)


def _system(lang: Lang, goal: str, tools: list[str], closing_rules: str) -> str:
    lines = "\n".join(localized(_TOOL_LINES[name], lang) for name in tools)
    return localized(_SYSTEM, lang).format(goal=goal, tools=lines, closing_rules=closing_rules)


def build_prioritizer_system_prompt(lang: Lang = Lang.PT, peer_requests: bool = False) -> str:
    """Prompt for prioritizing an item. `peer_requests` says whether the protocol gives the
    agent the action of asking the classifier to reconsider."""
    tools = ["recall_similar_cases", "backlog_overview"]
    if peer_requests:
        tools.append("request_reclassification")
    tools.append("submit_priority")
    return _system(lang, prioritizer_goal(lang), tools, localized(_FINAL_PRIORITY_RULES, lang))


def build_prioritizer_reconsideration_system_prompt(lang: Lang = Lang.PT) -> str:
    """Prompt for deciding again after the classifier's answer: no new request is possible,
    so the exchange cannot go on forever."""
    tools = ["recall_similar_cases", "backlog_overview", "submit_priority"]
    return _system(lang, reconsideration_goal(lang), tools, localized(_FINAL_PRIORITY_RULES, lang))


def build_prioritizer_reply_system_prompt(lang: Lang = Lang.PT) -> str:
    tools = ["recall_similar_cases", "backlog_overview", "reply_to_request"]
    return _system(lang, priority_reply_goal(lang), tools, localized(_REPLY_RULES, lang))


def describe_classification(
    requirement_type: str,
    nfr_category: str | None,
    confidence: float,
    justification: str,
    lang: Lang = Lang.PT,
) -> str:
    return localized(_REPORT, lang).format(
        requirement_type=requirement_type,
        category=nfr_category or "-",
        confidence=confidence,
        justification=justification or "-",
    )


def describe_priority(priority: str, score: float, justification: str, lang: Lang = Lang.PT) -> str:
    return localized(_PRIORITY, lang).format(
        priority=priority, score=score, justification=justification or "-"
    )


def describe_peer_answer(decision: str, reason: str, lang: Lang = Lang.PT) -> str:
    return localized(_PEER_ANSWER[decision], lang).format(reason=reason)


def build_prioritizer_percept(text: str, classification: str, lang: Lang = Lang.PT) -> str:
    return localized(_PERCEPT, lang).format(text=text, report=classification)


def build_reconsideration_percept(
    text: str, classification: str, previous: str, answer: str, lang: Lang = Lang.PT
) -> str:
    return localized(_RECONSIDERATION_PERCEPT, lang).format(
        text=text, report=classification, previous=previous, answer=answer
    )


def build_priority_reply_percept(
    text: str,
    classification: str,
    previous: str,
    reason: str,
    evidence: str,
    lang: Lang = Lang.PT,
) -> str:
    return localized(_REPLY_PERCEPT, lang).format(
        text=text,
        report=classification,
        previous=previous,
        reason=reason,
        evidence=evidence or "-",
    )
