from __future__ import annotations

from domain.enums import Lang

# ── Blocos de categorias NFR ───────────────────────────────────────────────────

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

# ── System prompts ─────────────────────────────────────────────────────────────

CLASSIFICATION_SYSTEM_PROMPT_PT = """\
Você é um especialista em Engenharia de Requisitos com profundo conhecimento \
no método de classificação funcional/não-funcional de requisitos de software.

Sua tarefa é classificar um requisito de software em:
1. TIPO: Funcional (F) ou Não-Funcional (NF)
2. CATEGORIA NFR (apenas se NF)

{nfr_block}

Regras obrigatórias:
- Responda SEMPRE em JSON válido, sem markdown.
- Se o tipo for F, o campo "nfr_category" deve ser null.
- "confidence" deve ser um float entre 0.0 e 1.0.
- "justification" deve ser concisa (máx. 2 frases), em português.

Formato de resposta:
{{
  "requirement_type": "F" | "NF",
  "nfr_category": "<categoria>" | null,
  "confidence": <float>,
  "justification": "<texto>"
}}
"""

CLASSIFICATION_SYSTEM_PROMPT_EN = """\
You are a Requirements Engineering expert with deep knowledge of \
functional/non-functional classification of software requirements.

Your task is to classify a software requirement as:
1. TYPE: Functional (F) or Non-Functional (NF)
2. NFR CATEGORY (only if NF)

{nfr_block}

Mandatory rules:
- Always respond with valid JSON, no markdown.
- If the type is F, "nfr_category" must be null.
- "confidence" must be a float between 0.0 and 1.0.
- "justification" must be concise (max 2 sentences), in English.

Response format:
{{
  "requirement_type": "F" | "NF",
  "nfr_category": "<category>" | null,
  "confidence": <float>,
  "justification": "<text>"
}}
"""


CLASSIFICATION_USER_PROMPT_PT = """\
Classifique o seguinte requisito de software:

\"\"\"{requirement_text}\"\"\"
"""

CLASSIFICATION_USER_PROMPT_EN = """\
Classify the following software requirement:

\"\"\"{requirement_text}\"\"\"
"""


def _build_nfr_block(
    nfr_categories: list[tuple[str, str]] | None,
    lang: Lang,
) -> str:
    """Renderiza o bloco de categorias NFR no prompt de forma condicional."""
    if nfr_categories is None:
        return _NFR_BLOCK_GENERIC_PT if lang is Lang.PT else _NFR_BLOCK_GENERIC_EN

    categories_str = "\n".join(f"  - {code}: {desc}" for code, desc in nfr_categories)
    template = _NFR_BLOCK_PT if lang is Lang.PT else _NFR_BLOCK_EN
    return template.format(categories=categories_str)


def build_classification_messages(
    requirement_text: str,
    lang: Lang = Lang.PT,
    nfr_categories: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Constrói mensagens para a chamada LLM do classificador.

    Args:
        requirement_text: Texto do requisito a ser classificado.
        lang: Lang.PT (português) ou Lang.EN (inglês).
        nfr_categories: Lista de (código, descrição) das categorias NFR do dataset.
                        None = sem taxonomia estruturada (prompt genérico).
    """
    nfr_block = _build_nfr_block(nfr_categories, lang)

    system_template = (
        CLASSIFICATION_SYSTEM_PROMPT_PT if lang is Lang.PT else CLASSIFICATION_SYSTEM_PROMPT_EN
    )
    system = system_template.format(nfr_block=nfr_block)

    user_template = (
        CLASSIFICATION_USER_PROMPT_PT if lang is Lang.PT else CLASSIFICATION_USER_PROMPT_EN
    )
    user = user_template.format(requirement_text=requirement_text)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
