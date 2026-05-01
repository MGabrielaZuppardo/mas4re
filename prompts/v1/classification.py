from domain.enums import NFRCategory

_CATEGORIES_LIST = "\n".join(
    f"  - {c.value}: {c.name.replace('_', ' ').title()}" for c in NFRCategory
)

CLASSIFICATION_SYSTEM_PROMPT = f"""\
Você é um especialista em Engenharia de Requisitos com profundo conhecimento \
no dataset PROMISE NFR (Cleland-Huang et al., 2007).

Sua tarefa é classificar um requisito de software em:
1. TIPO: Funcional (F) ou Não-Funcional (NF)
2. CATEGORIA NFR (apenas se NF): uma das categorias abaixo

Categorias NFR disponíveis:
{_CATEGORIES_LIST}

Regras obrigatórias:
- Responda SEMPRE em JSON válido, sem markdown.
- Se o tipo for F, o campo "nfr_category" deve ser null.
- "confidence" deve ser um float entre 0.0 e 1.0.
- "justification" deve ser concisa (máx. 2 frases).

Formato de resposta:
{{
  "requirement_type": "F" | "NF",
  "nfr_category": "<SIGLA>" | null,
  "confidence": <float>,
  "justification": "<texto>"
}}
"""

CLASSIFICATION_USER_PROMPT = """\
Classifique o seguinte requisito de software:

\"\"\"{requirement_text}\"\"\"
"""


def build_classification_messages(requirement_text: str) -> list[dict[str, str]]:
    """Constrói mensagens para a chamada LLM."""
    return [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": CLASSIFICATION_USER_PROMPT.format(requirement_text=requirement_text),
        },
    ]
