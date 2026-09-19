from __future__ import annotations

import json
from collections.abc import Mapping

from domain.enums import Lang
from prompts.v1._shared import build_nfr_block

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


# ── Self-critique prompts (ADR-011, Madaan et al. 2023 Self-Refine) ────────────

CLASSIFICATION_CRITIQUE_SYSTEM_PROMPT_PT = """\
Você é um revisor especialista em Engenharia de Requisitos. Sua tarefa é revisar uma \
classificação funcional/não-funcional já feita por outro especialista e decidir se ela precisa \
de correção.

Responda SEMPRE em JSON válido, sem markdown:
{{
  "needs_revision": true | false,
  "issue": "<o que está errado, ou string vazia se não precisar de revisão>",
  "revised_output": {{
    "requirement_type": "F" | "NF",
    "nfr_category": "<categoria>" | null,
    "confidence": <float 0.0-1.0>,
    "justification": "<texto>"
  }} | null
}}

Se "needs_revision" for false, "revised_output" deve ser null.
"""

CLASSIFICATION_CRITIQUE_SYSTEM_PROMPT_EN = """\
You are a Requirements Engineering expert reviewer. Your task is to review a \
functional/non-functional classification already made by another expert and decide whether it \
needs correction.

Always respond with valid JSON, no markdown:
{{
  "needs_revision": true | false,
  "issue": "<what is wrong, or empty string if no revision is needed>",
  "revised_output": {{
    "requirement_type": "F" | "NF",
    "nfr_category": "<category>" | null,
    "confidence": <float 0.0-1.0>,
    "justification": "<text>"
  }} | null
}}

If "needs_revision" is false, "revised_output" must be null.
"""

CLASSIFICATION_CRITIQUE_USER_PROMPT_PT = """\
Requisito: \"\"\"{requirement_text}\"\"\"

Classificação a revisar:
{original_answer_json}
"""

CLASSIFICATION_CRITIQUE_USER_PROMPT_EN = """\
Requirement: \"\"\"{requirement_text}\"\"\"

Classification to review:
{original_answer_json}
"""


def build_classification_critique_messages(
    requirement_text: str,
    original_answer: Mapping[str, object],
    lang: Lang = Lang.PT,
) -> list[dict[str, str]]:
    """Constrói mensagens para a rodada de autocrítica (Self-Refine).

    original_answer: os campos de ClassificationOutput já gerados
    (requirement_type/nfr_category/confidence/justification), como dict.
    """
    is_pt = lang is Lang.PT
    system = (
        CLASSIFICATION_CRITIQUE_SYSTEM_PROMPT_PT
        if is_pt
        else CLASSIFICATION_CRITIQUE_SYSTEM_PROMPT_EN
    )
    template = (
        CLASSIFICATION_CRITIQUE_USER_PROMPT_PT if is_pt else CLASSIFICATION_CRITIQUE_USER_PROMPT_EN
    )
    user = template.format(
        requirement_text=requirement_text,
        original_answer_json=json.dumps(original_answer, ensure_ascii=False),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_classification_messages(
    requirement_text: str,
    lang: Lang = Lang.PT,
    nfr_categories: list[tuple[str, str]] | None = None,
    few_shot_block: str = "",
) -> list[dict[str, str]]:
    """Constrói mensagens para a chamada LLM do classificador.

    Args:
        requirement_text: Texto do requisito a ser classificado.
        lang: Lang.PT (português) ou Lang.EN (inglês).
        nfr_categories: Lista de (código, descrição) das categorias NFR do dataset.
                        None = sem taxonomia estruturada (prompt genérico).
        few_shot_block: Bloco opcional de exemplos já processados no lote
                        (memória episódica, ADR-011), prefixado à mensagem
                        de usuário. Vazio por padrão (sem memória).
    """
    nfr_block = build_nfr_block(nfr_categories, lang)

    system_template = (
        CLASSIFICATION_SYSTEM_PROMPT_PT if lang is Lang.PT else CLASSIFICATION_SYSTEM_PROMPT_EN
    )
    system = system_template.format(nfr_block=nfr_block)

    user_template = (
        CLASSIFICATION_USER_PROMPT_PT if lang is Lang.PT else CLASSIFICATION_USER_PROMPT_EN
    )
    user = few_shot_block + user_template.format(requirement_text=requirement_text)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
