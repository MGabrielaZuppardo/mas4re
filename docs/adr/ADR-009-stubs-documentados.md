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
```

## Nota de Atualização — 2026-07-12

Adoção mista:

- Vários stubs deixaram de ser stub e foram totalmente implementados:
  `pipeline/graph.py`, `pipeline/nodes/classifier_node.py`,
  `pipeline/nodes/prioritizer_node.py`, `cli/main.py`.
- `pipeline/nodes/elicitor_node.py` segue corretamente o padrão de
  placeholder documentado (docstring multi-linha explicando o escopo
  fora do MAS4RE v1).
- Porém vários arquivos continuam **0 bytes**, violando a própria
  política desta ADR: `agents/elicitor.py`, `datasets/nfric.py`,
  `config/registry.py`, `pipeline/nodes/__init__.py`,
  `datasets/__init__.py`, `evaluation/metrics/__init__.py`,
  `config/__init__.py` (verificar também `prompts/v1/elicitation.py`).

Recomenda-se um commit dedicado para aplicar o placeholder de uma linha
aos arquivos ainda vazios, fechando o gap entre política e prática.
