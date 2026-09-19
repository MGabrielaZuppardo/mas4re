from __future__ import annotations

import json
from collections.abc import Mapping

from domain.enums import Lang

PRIORITIZATION_SYSTEM_PROMPT_PT = """\
Você é um especialista em Engenharia de Requisitos com profundo conhecimento \
no método de priorização MoSCoW (Clegg & Barker, 1994).

Sua tarefa é priorizar um requisito de software usando o método MoSCoW:
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

Regras obrigatórias:
- Responda SEMPRE em JSON válido, sem markdown.
- "priority_score" deve ser float entre 0.0 e 1.0 refletindo a urgência.
  Referência: M=1.0, S=0.75, C=0.5, W=0.25 (variações dentro da faixa são encorajadas).
- "priority_rank" deve ser 1 (será recalculado globalmente pelo sistema).
- "justification" deve ser concisa (máx. 2 frases), em português.
- Se "classification_confidence" for inferior a 0.70, a classificação do tipo é incerta.
  Nesse caso, seja conservador: evite M para requisitos borderline e prefira S ou C.

Formato de resposta:
{
  "priority": "M" | "S" | "C" | "W",
  "priority_score": <float 0.0–1.0>,
  "priority_rank": 1,
  "justification": "<texto>"
}
"""

PRIORITIZATION_SYSTEM_PROMPT_EN = """\
You are a Requirements Engineering expert with deep knowledge of the MoSCoW \
prioritization method (Clegg & Barker, 1994).

Your task is to prioritize a software requirement using MoSCoW:
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

Mandatory rules:
- Always respond with valid JSON, no markdown.
- "priority_score" must be a float 0.0–1.0 reflecting urgency.
  Reference: M=1.0, S=0.75, C=0.5, W=0.25 (variations within range are encouraged).
- "priority_rank" must be 1 (will be recalculated globally by the system).
- "justification" must be concise (max 2 sentences), in English.
- If "classification_confidence" is below 0.70, the requirement type is uncertain.
  In that case, be conservative: avoid M for borderline requirements and prefer S or C.

Response format:
{
  "priority": "M" | "S" | "C" | "W",
  "priority_score": <float 0.0–1.0>,
  "priority_rank": 1,
  "justification": "<text>"
}
"""

PRIORITIZATION_USER_PROMPT_PT = """\
Priorize o seguinte requisito de software:

Texto: \"\"\"{requirement_text}\"\"\"
Tipo: {requirement_type}{nfr_category_line}
classification_confidence: {confidence:.2f}{uncertainty_flag}
"""

PRIORITIZATION_USER_PROMPT_EN = """\
Prioritize the following software requirement:

Text: \"\"\"{requirement_text}\"\"\"
Type: {requirement_type}{nfr_category_line}
classification_confidence: {confidence:.2f}{uncertainty_flag}
"""

_UNCERTAINTY_FLAG_PT = " (INCERTA — a classificação pode estar incorreta)"
_UNCERTAINTY_FLAG_EN = " (UNCERTAIN — classification may be incorrect)"


# ── Self-critique prompts (ADR-011, Madaan et al. 2023 Self-Refine) ────────────

PRIORITIZATION_CRITIQUE_SYSTEM_PROMPT_PT = """\
Você é um revisor especialista em Engenharia de Requisitos. Sua tarefa é revisar uma \
priorização MoSCoW já feita por outro especialista e decidir se ela precisa de correção.

Responda SEMPRE em JSON válido, sem markdown:
{{
  "needs_revision": true | false,
  "issue": "<o que está errado, ou string vazia se não precisar de revisão>",
  "revised_output": {{
    "priority": "M" | "S" | "C" | "W",
    "priority_score": <float 0.0-1.0>,
    "priority_rank": 1,
    "justification": "<texto>"
  }} | null
}}

Se "needs_revision" for false, "revised_output" deve ser null.
"""

PRIORITIZATION_CRITIQUE_SYSTEM_PROMPT_EN = """\
You are a Requirements Engineering expert reviewer. Your task is to review a MoSCoW \
prioritization already made by another expert and decide whether it needs correction.

Always respond with valid JSON, no markdown:
{{
  "needs_revision": true | false,
  "issue": "<what is wrong, or empty string if no revision is needed>",
  "revised_output": {{
    "priority": "M" | "S" | "C" | "W",
    "priority_score": <float 0.0-1.0>,
    "priority_rank": 1,
    "justification": "<text>"
  }} | null
}}

If "needs_revision" is false, "revised_output" must be null.
"""

PRIORITIZATION_CRITIQUE_USER_PROMPT_PT = """\
Requisito: \"\"\"{requirement_text}\"\"\"

Priorização a revisar:
{original_answer_json}
"""

PRIORITIZATION_CRITIQUE_USER_PROMPT_EN = """\
Requirement: \"\"\"{requirement_text}\"\"\"

Prioritization to review:
{original_answer_json}
"""


def build_prioritization_critique_messages(
    requirement_text: str,
    original_answer: Mapping[str, object],
    lang: Lang = Lang.PT,
) -> list[dict[str, str]]:
    """Constrói mensagens para a rodada de autocrítica (Self-Refine).

    original_answer: os campos de PrioritizationOutput já gerados
    (priority/priority_score/priority_rank/justification), como dict.
    """
    is_pt = lang is Lang.PT
    system = (
        PRIORITIZATION_CRITIQUE_SYSTEM_PROMPT_PT
        if is_pt
        else PRIORITIZATION_CRITIQUE_SYSTEM_PROMPT_EN
    )
    template = (
        PRIORITIZATION_CRITIQUE_USER_PROMPT_PT if is_pt else PRIORITIZATION_CRITIQUE_USER_PROMPT_EN
    )
    user = template.format(
        requirement_text=requirement_text,
        original_answer_json=json.dumps(original_answer, ensure_ascii=False),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_prioritization_messages(
    requirement_text: str,
    requirement_type: str,
    nfr_category: str | None = None,
    confidence: float = 1.0,
    lang: Lang = Lang.PT,
    few_shot_block: str = "",
) -> list[dict[str, str]]:
    """Constrói mensagens para a chamada LLM do agente de priorização.

    Args:
        requirement_text: Texto do requisito.
        requirement_type: 'F' ou 'NF'.
        nfr_category: Sigla da categoria NFR (ex: 'SE', 'PE') ou None.
        confidence: Confiança do ClassificationAgent (0.0–1.0). Abaixo de
            0.70 sinaliza classificação incerta ao modelo prioritizador.
        lang: Lang.PT (português) ou Lang.EN (inglês).
        few_shot_block: Bloco opcional de exemplos já processados no lote
                        (memória episódica, ADR-011). Vazio por padrão.
    """
    is_pt = lang is Lang.PT
    system = PRIORITIZATION_SYSTEM_PROMPT_PT if is_pt else PRIORITIZATION_SYSTEM_PROMPT_EN
    template = PRIORITIZATION_USER_PROMPT_PT if is_pt else PRIORITIZATION_USER_PROMPT_EN

    nfr_line = (
        (f"\nCategoria NFR: {nfr_category}" if is_pt else f"\nNFR Category: {nfr_category}")
        if nfr_category
        else ""
    )
    uncertainty_flag = (
        (_UNCERTAINTY_FLAG_PT if is_pt else _UNCERTAINTY_FLAG_EN) if confidence < 0.70 else ""
    )

    user = few_shot_block + template.format(
        requirement_text=requirement_text,
        requirement_type=requirement_type,
        nfr_category_line=nfr_line,
        confidence=confidence,
        uncertainty_flag=uncertainty_flag,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
