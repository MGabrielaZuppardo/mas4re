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
```

## Nota de Atualização — 2026-07-12

Auditoria de código mostra que esta decisão **não foi adotada na prática**.
`agents/base.py.__init__` ainda recebe apenas `model: str, temperature: float,
trace_writer`, sem `llm`/`prompt_builder`/`settings` injetados. Todos os
agentes concretos continuam construindo o LLM internamente:

- `agents/baseline.py:26,45` — `from llm.factory import build_llm` /
  `self._llm = build_llm(model, temperature)`
- `agents/classifier.py:26,45` — mesmo padrão
- `agents/prioritizer.py:26,54` — mesmo padrão
- `agents/two_call_baseline.py:60,193` — mesmo padrão (agente introduzido
  após esta ADR, já nasceu sem DI)

O acoplamento a `llm/factory.py` que a ADR buscava eliminar permanece
presente em 100% dos agentes. Status de implementação: **não adotada** —
decisão continua válida como direção arquitetural, mas requer trabalho
futuro para ser efetivada.
