# ADR-000: Adoção de Architecture Decision Records

- **Status:** Accepted
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** infra

## Contexto

O projeto é uma dissertação de mestrado com três perguntas de pesquisa
(SQ1, SQ2, SQ3) e múltiplas decisões técnicas não-óbvias que afetam a
validade dos experimentos. Sem registro versionado, essas decisões
desaparecem em Slack, e-mail ou conversa com orientador, dificultando:
- defesa científica das escolhas perante a banca,
- onboarding de colaboradores,
- reversão consciente de decisões anteriores.

## Decisão

Adotar Architecture Decision Records (ADRs) leves, em `docs/adr/`, como
mecanismo oficial de registro de decisões arquiteturais. Cada ADR é um
arquivo Markdown imutável após `Accepted`. Mudanças posteriores entram
via novo ADR que marca o anterior como `Superseded by`.

## Consequências

**Positivas:**
- Trilha de auditoria para decisões , útil na banca e em revisões.
- Reduz "tribal knowledge" , quem chega lê a pasta e entende o porquê.
- ADRs podem ser citados em PRs, código e capítulos da dissertação.

**Negativas / trade-offs aceitos:**
- Overhead de escrita: cada decisão consome ~15–30 min.
- Risco de virar burocracia se aplicado a decisões triviais.

**Neutras:**
- Não exige ferramentas; basta um editor de Markdown.

## Alternativas consideradas

- **Wiki externa (Notion, Confluence):** descartada — separa decisão do
  código, sofre rot, não versiona com o repositório.
- **Apenas mensagens de commit:** descartada — commits descrevem
  o "como", não capturam o "por quê" estruturado.

## Impacto na pesquisa

Habilita citação direta de decisões na dissertação ("conforme ADR-005,
optou-se por LangGraph..."), fortalecendo a defesa metodológica.

## Referências
- [`_template.md`](_template.md)
- docs/refactor-plan.md