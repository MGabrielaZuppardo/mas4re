from __future__ import annotations

import json
from collections.abc import Mapping

from domain.enums import Lang
from prompts.localization import localized
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
classificação funcional/não-funcional já feita por outro especialista e decidir se ela contém \
um erro CLARO.

Critérios de referência:
- Funcional (F): descreve o que o sistema deve FAZER (funções, comportamentos, dados, interações).
- Não-Funcional (NF): descreve uma QUALIDADE ou RESTRIÇÃO sobre como o sistema opera.

{nfr_block}

Revise somente se a classificação contrariar claramente os critérios acima; na dúvida, mantenha-a.

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
contains a CLEAR error.

Reference criteria:
- Functional (F): describes what the system must DO (functions, behaviors, data, interactions).
- Non-Functional (NF): describes a QUALITY or CONSTRAINT on how the system operates.

{nfr_block}

Revise only if the classification clearly contradicts the criteria above; when in doubt, keep it.

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


CLASSIFICATION_SYSTEM_PROMPT_ES = """\
Eres un especialista en Ingeniería de Requisitos con profundo conocimiento \
del método de clasificación funcional/no funcional de requisitos de software.

Tu tarea es clasificar un requisito de software en:
1. TIPO: Funcional (F) o No Funcional (NF)
2. CATEGORÍA NFR (solo si es NF)

{nfr_block}

Reglas obligatorias:
- Responde SIEMPRE en JSON válido, sin markdown.
- Si el tipo es F, el campo "nfr_category" debe ser null.
- "confidence" debe ser un número decimal entre 0.0 y 1.0.
- "justification" debe ser concisa (máx. 2 frases), en español.

Formato de respuesta:
{{
  "requirement_type": "F" | "NF",
  "nfr_category": "<categoría>" | null,
  "confidence": <decimal>,
  "justification": "<texto>"
}}
"""

CLASSIFICATION_USER_PROMPT_ES = """\
Clasifica el siguiente requisito de software:

\"\"\"{requirement_text}\"\"\"
"""

CLASSIFICATION_CRITIQUE_SYSTEM_PROMPT_ES = """\
Eres un revisor especialista en Ingeniería de Requisitos. Tu tarea es revisar una \
clasificación funcional/no funcional ya realizada por otro especialista y decidir si contiene \
un error CLARO.

Criterios de referencia:
- Funcional (F): describe lo que el sistema debe HACER (funciones, comportamientos, datos, \
interacciones).
- No Funcional (NF): describe una CUALIDAD o RESTRICCIÓN sobre cómo opera el sistema.

{nfr_block}

Revisa solo si la clasificación contradice claramente los criterios anteriores; ante la duda, \
mantenla.

Responde SIEMPRE en JSON válido, sin markdown:
{{
  "needs_revision": true | false,
  "issue": "<qué está mal, o cadena vacía si no necesita revisión>",
  "revised_output": {{
    "requirement_type": "F" | "NF",
    "nfr_category": "<categoría>" | null,
    "confidence": <decimal 0.0-1.0>,
    "justification": "<texto>"
  }} | null
}}

Si "needs_revision" es false, "revised_output" debe ser null.
"""

CLASSIFICATION_CRITIQUE_USER_PROMPT_ES = """\
Requisito: \"\"\"{requirement_text}\"\"\"

Clasificación a revisar:
{original_answer_json}
"""

_SYSTEM: dict[Lang, str] = {
    Lang.PT: CLASSIFICATION_SYSTEM_PROMPT_PT,
    Lang.EN: CLASSIFICATION_SYSTEM_PROMPT_EN,
    Lang.ES: CLASSIFICATION_SYSTEM_PROMPT_ES,
}
_USER: dict[Lang, str] = {
    Lang.PT: CLASSIFICATION_USER_PROMPT_PT,
    Lang.EN: CLASSIFICATION_USER_PROMPT_EN,
    Lang.ES: CLASSIFICATION_USER_PROMPT_ES,
}
_CRITIQUE_SYSTEM: dict[Lang, str] = {
    Lang.PT: CLASSIFICATION_CRITIQUE_SYSTEM_PROMPT_PT,
    Lang.EN: CLASSIFICATION_CRITIQUE_SYSTEM_PROMPT_EN,
    Lang.ES: CLASSIFICATION_CRITIQUE_SYSTEM_PROMPT_ES,
}
_CRITIQUE_USER: dict[Lang, str] = {
    Lang.PT: CLASSIFICATION_CRITIQUE_USER_PROMPT_PT,
    Lang.EN: CLASSIFICATION_CRITIQUE_USER_PROMPT_EN,
    Lang.ES: CLASSIFICATION_CRITIQUE_USER_PROMPT_ES,
}


def build_classification_critique_messages(
    requirement_text: str,
    original_answer: Mapping[str, object],
    lang: Lang = Lang.PT,
    nfr_categories: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Constrói mensagens para a rodada de autocrítica (Self-Refine).

    original_answer: os campos de ClassificationOutput já gerados
    (requirement_type/nfr_category/confidence/justification), como dict.
    nfr_categories: taxonomia do dataset, o padrão externo contra o qual o
        crítico julga (mesmo bloco usado no prompt de classificação).
    """
    system = localized(_CRITIQUE_SYSTEM, lang).format(
        nfr_block=build_nfr_block(nfr_categories, lang)
    )
    user = localized(_CRITIQUE_USER, lang).format(
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
        lang: Idioma do prompt (Lang.PT, Lang.EN ou Lang.ES).
        nfr_categories: Lista de (código, descrição) das categorias NFR do dataset.
                        None = sem taxonomia estruturada (prompt genérico).
        few_shot_block: Bloco opcional de exemplos já processados no lote
                        (memória episódica, ADR-011), prefixado à mensagem
                        de usuário. Vazio por padrão (sem memória).
    """
    nfr_block = build_nfr_block(nfr_categories, lang)

    system = localized(_SYSTEM, lang).format(nfr_block=nfr_block)
    user = few_shot_block + localized(_USER, lang).format(requirement_text=requirement_text)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
