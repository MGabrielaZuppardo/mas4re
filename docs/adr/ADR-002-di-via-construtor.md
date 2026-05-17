# ADR-002: Injeção de dependências via construtor nos agentes

- **Status:** Accepted
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** infra

## Contexto

Agentes instanciam `build_llm(model, temperature)` no `__init__`
(ex.: `agents/baseline.py:49`). Isso acopla o agente ao módulo
`llm/factory.py` e a `Settings`. Consequências:
- testes precisam mockar import de `build_llm`,
- trocar provedor de LLM exige editar agente,
- viola inversão de dependência.

## Decisão

Agentes recebem dependências no construtor:

```python
class BaseAgent:
    def __init__(self, llm: BaseChatModel, prompt_builder: PromptBuilder,
                 settings: Settings, trace_writer: TraceWriter | None = None): ...
