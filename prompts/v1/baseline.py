from __future__ import annotations

# ── Categorias NFR 

_NFR_BLOCK_PT = """\
Se NF, atribua uma das categorias abaixo:
{categories}"""

_NFR_BLOCK_GENERIC_PT = """\
Se NF, descreva a categoria com uma palavra-chave concisa \
(ex: "desempenho", "segurança", "usabilidade")."""

_NFR_BLOCK_EN = """\
If NF, assign one of the following subcategories:
{categories}"""

_NFR_BLOCK_GENERIC_EN = """\
If NF, describe the category with a concise keyword \
(e.g., "performance", "security", "usability")."""

# ── System prompts 

BASELINE_SYSTEM_PROMPT_PT = """\
Você é um especialista em Engenharia de Requisitos com profundo conhecimento \
no método de classificação funcional/não-funcional e no método de \
priorização MoSCoW (Clegg & Barker, 1994).

Sua tarefa é, em uma única análise, classificar E priorizar um requisito \
de software.

─── ETAPA 1 · CLASSIFICAÇÃO ──────────────────────────────────────────────

Classifique o requisito em:
· Funcional (F): descreve o que o sistema deve fazer.
· Não-Funcional (NF): descreve uma qualidade ou restrição do sistema.

{nfr_block}

─── ETAPA 2 · PRIORIZAÇÃO ────────────────────────────────────────────────

Priorize o requisito usando MoSCoW:
· M (Must Have)   — Obrigatório. Sem isso o sistema não funciona.
· S (Should Have) — Importante, mas viável adiar.
· C (Could Have)  — Desejável, pode ser excluído sem grande impacto.
· W (Won't Have)  — Fora do escopo atual.

Critérios por tipo:
· F:  avalie impacto no negócio, frequência de uso e dependências.
· NF: avalie risco operacional, conformidade legal e experiência do usuário.

─── REGRAS ────────────────────────────────────────────────────────────────

- Responda SEMPRE em JSON válido, sem markdown.
- Se o tipo for F, "nfr_category" deve ser null.
- "confidence" e "priority_score" são floats entre 0.0 e 1.0.
  Referência de score: M=1.0, S=0.75, C=0.5, W=0.25.
- "priority_rank" deve ser 1 (será recalculado globalmente pelo sistema).
- As justificativas devem ser concisas (máx. 2 frases cada), em português.

─── FORMATO DE RESPOSTA ──────────────────────────────────────────────────

{{
  "requirement_type": "F" | "NF",
  "nfr_category": "<categoria>" | null,
  "confidence": <float>,
  "classification_justification": "<texto>",
  "priority": "M" | "S" | "C" | "W",
  "priority_score": <float>,
  "priority_rank": 1,
  "priority_justification": "<texto>"
}}
"""

BASELINE_SYSTEM_PROMPT_EN = """\
You are a Requirements Engineering expert with deep knowledge of \
functional/non-functional classification and the MoSCoW \
prioritization method (Clegg & Barker, 1994).

Your task is to classify AND prioritize a software requirement in a \
single analysis.

─── STEP 1 · CLASSIFICATION ──────────────────────────────────────────────

Classify the requirement as:
· Functional (F): describes what the system must do.
· Non-Functional (NF): describes a quality attribute or constraint.

{nfr_block}

─── STEP 2 · PRIORITIZATION ──────────────────────────────────────────────

Prioritize the requirement using MoSCoW:
· M (Must Have)   — Critical. The system cannot function without it.
· S (Should Have) — Important but not critical. High priority, deferrable.
· C (Could Have)  — Desirable. Adds value but removable with minor impact.
· W (Won't Have)  — Out of scope for now.

Criteria by type:
· F:  evaluate business impact, usage frequency and dependencies.
· NF: evaluate operational risk, legal compliance and user experience.

─── RULES ────────────────────────────────────────────────────────────────

- Always respond with valid JSON, no markdown.
- If the type is F, "nfr_category" must be null.
- "confidence" and "priority_score" are floats between 0.0 and 1.0.
  Score reference: M=1.0, S=0.75, C=0.5, W=0.25.
- "priority_rank" must be 1 (will be recalculated globally by the system).
- Justifications must be concise (max 2 sentences each), in English.

─── RESPONSE FORMAT ──────────────────────────────────────────────────────

{{
  "requirement_type": "F" | "NF",
  "nfr_category": "<category>" | null,
  "confidence": <float>,
  "classification_justification": "<text>",
  "priority": "M" | "S" | "C" | "W",
  "priority_score": <float>,
  "priority_rank": 1,
  "priority_justification": "<text>"
}}
"""

# ── User prompts 

BASELINE_USER_PROMPT_PT = """\
Classifique e priorize o seguinte requisito de software:

\"\"\"{requirement_text}\"\"\"
"""

BASELINE_USER_PROMPT_EN = """\
Classify and prioritize the following software requirement:

\"\"\"{requirement_text}\"\"\"
"""

# ── Builders 

def _build_nfr_block(
    nfr_categories: list[tuple[str, str]] | None,
    lang: str,
) -> str:
    """Renderiza o bloco de categorias NFR no prompt de forma condicional."""
    if nfr_categories is None:
        return _NFR_BLOCK_GENERIC_PT if lang == "pt" else _NFR_BLOCK_GENERIC_EN

    categories_str = "\n".join(
        f"  - {code}: {desc}" for code, desc in nfr_categories
    )
    template = _NFR_BLOCK_PT if lang == "pt" else _NFR_BLOCK_EN
    return template.format(categories=categories_str)


def build_baseline_messages(
    requirement_text: str,
    lang: str = "pt",
    nfr_categories: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Constrói mensagens para a chamada LLM do agente baseline.

    Args:
        requirement_text: Texto do requisito a ser processado.
        lang: 'pt' para prompt em português, 'en' para inglês.
        nfr_categories: Lista de (código, descrição) das categorias NFR do dataset.
                        None = sem taxonomia estruturada (prompt genérico).
    """
    nfr_block = _build_nfr_block(nfr_categories, lang)

    system_template = (
        BASELINE_SYSTEM_PROMPT_PT if lang == "pt" else BASELINE_SYSTEM_PROMPT_EN
    )
    system = system_template.format(nfr_block=nfr_block)

    user_template = (
        BASELINE_USER_PROMPT_PT if lang == "pt" else BASELINE_USER_PROMPT_EN
    )
    user = user_template.format(requirement_text=requirement_text)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]