from __future__ import annotations

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

Response format:
{
  "priority": "M" | "S" | "C" | "W",
  "priority_score": <float 0.0–1.0>,
  "priority_rank": 1,
  "justification": "<text>"
}
"""

PRIORITIZATION_USER_PROMPT = """\
Priorize o seguinte requisito de software:

Texto: \"\"\"{requirement_text}\"\"\"
Tipo: {requirement_type}{nfr_category_line}
"""


def build_prioritization_messages(
    requirement_text: str,
    requirement_type: str,
    nfr_category: str | None = None,
    lang: Lang = Lang.PT,
) -> list[dict[str, str]]:
    """
    Constrói mensagens para a chamada LLM.

    Args:
        requirement_text: Texto do requisito.
        requirement_type: 'F' ou 'NF'.
        nfr_category: Sigla da categoria NFR (ex: 'SE', 'PE') ou None.
        lang: Lang.PT (português) ou Lang.EN (inglês).
    """
    system = PRIORITIZATION_SYSTEM_PROMPT_PT if lang is Lang.PT else PRIORITIZATION_SYSTEM_PROMPT_EN
    nfr_line = f"\nCategoria NFR: {nfr_category}" if nfr_category else ""
    user = PRIORITIZATION_USER_PROMPT.format(
        requirement_text=requirement_text,
        requirement_type=requirement_type,
        nfr_category_line=nfr_line,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
