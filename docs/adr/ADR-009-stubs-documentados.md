# ADR-009: Stubs vazios viram placeholders documentados

- **Status:** Accepted
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** infra

## Contexto

O repositório tem 12 arquivos `.py` com 0 linhas (`pipeline/graph.py`,
`pipeline/nodes/*.py`, `cli/main.py`, `agents/elicitor.py`, etc.).
Empty `.py` enganam grep, IDE marca como problema, e contradizem a
arquitetura prometida. Por outro lado, apagar e recriar gera churn
no git e perde a "promessa estrutural".

## Decisão

Stubs vazios viram **placeholders de uma linha**:

```python
"""Reserved: implementação na Semana X, Feature Y.Z — ver docs/adr/ADR-NNN."""
