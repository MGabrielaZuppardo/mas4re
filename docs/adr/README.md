# Architecture Decision Records — mas4re

Registro de decisões arquiteturais. Cada ADR é imutável após `Accepted`;
mudanças posteriores entram via novo ADR que marca o anterior como `Superseded by`.

| #   | Título                                                | Status   | Data       | SQ   |
|-----|-------------------------------------------------------|----------|------------|------|
| 000 | [Adoção de ADRs](ADR-000-adocao-de-adrs.md)           | Accepted | YYYY-MM-DD | infra |
| 001 | [BaseAgent genérico](ADR-001-base-agent-generic.md)   | Proposed | YYYY-MM-DD | infra |
| 002 | [DI via construtor](ADR-002-di-via-construtor.md)     | Proposed | YYYY-MM-DD | infra |
| 003 | [Taxonomia de falhas SQ3](ADR-003-failure-taxonomy.md)| Proposed | YYYY-MM-DD | SQ3  |
| 004 | [Strategy + Runner](ADR-004-strategy-runner.md)       | Proposed | YYYY-MM-DD | SQ2  |
| 005 | [Orquestração LangGraph](ADR-005-langgraph.md)        | Proposed | YYYY-MM-DD | SQ2  |
| 006 | [Logging estruturado](ADR-006-logging.md)             | Proposed | YYYY-MM-DD | infra |
| 007 | [Idiomas EN/PT](ADR-007-idiomas.md)                   | Proposed | YYYY-MM-DD | infra |
| 008 | [Credenciais condicionais](ADR-008-credenciais.md)    | Proposed | YYYY-MM-DD | infra |
| 009 | [Stubs documentados](ADR-009-stubs-documentados.md)   | Proposed | YYYY-MM-DD | infra |
| 010 | [Elicitor: anota vs modifica](ADR-010-elicitor.md)    | Proposed | YYYY-MM-DD | SQ1/SQ3 |

## Convenções
- **Nome do arquivo:** `ADR-NNN-slug-kebab-case.md`
- **Numeração:** sequencial, nunca reusada
- **Imutabilidade:** ADR aceito não se edita; cria-se um novo que o supersede
- **Template:** [`_template.md`](_template.md)