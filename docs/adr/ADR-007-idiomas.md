# ADR-007: Política de idiomas — EN no código, PT na documentação

- **Status:** Accepted
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** infra

## Contexto

O código mistura PT e EN sem critério: docstrings PT, identificadores
EN, enums com sufixos PT ("label_pt"), parâmetro `lang: str = "pt"`
sem enum. Inconsistência confunde leitor e gera bugs (string "pt" vs
"PT"). Para a dissertação, docs precisam estar em PT.

## Decisão

| Camada | Idioma |
|---|---|
| Identificadores Python | EN |
| Docstrings e comentários | EN |
| Mensagens de log | EN |
| Output CLI ao usuário | EN ou PT (via i18n simples) |
| Prompts ao LLM | PT ou EN via `Lang(StrEnum): PT, EN` |
| Documentação para banca | PT |
| Glossário PT↔EN | `docs/glossary.md` |

Renomes: `justification_priority` → `priority_justification`
(sufixo consistente).

## Consequências

**Positivas:**
- Alcance internacional do código sem prejudicar a dissertação.
- Enum `Lang` impede typos ("pt" vs "PT").
- Glossário evita ambiguidade na banca.

**Negativas:**
- Renomes pontuais em `domain/models.py`.

**Neutras:**
- Política precisa ser aplicada em revisões de PR.

## Alternativas consideradas

- **100% PT:** descartada — fecha o código a colaboração externa.
- **100% EN inclusive docs da dissertação:** descartada — exigência
  da banca/UFPE.

## Impacto na pesquisa

Glossário evita confusão de terminologia (NFR, MoSCoW, etc.) entre
código e capítulos da dissertação.

## Referências
- `domain/enums.py`, `domain/models.py`, `docs/glossary.md`