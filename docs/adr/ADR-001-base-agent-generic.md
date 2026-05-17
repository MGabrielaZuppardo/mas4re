# ADR-001: BaseAgent genérico via Generic[Item, Output]

- **Status:** Accepted
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** infra (habilita SQ1, SQ2, SQ3)

## Contexto

`agents/base.py:11–14` declara `run(state: Any) -> Any` e
`_process_single(item: Any) -> Any`. Sem contrato de tipo, mypy não
verifica subclasses; bugs viajam silenciosamente entre `ClassificationAgent`,
`PrioritizationAgent` e `BaselineAgent`.

O pipeline deve aceitar **datasets diferentes** (PROMISE, NFRic, futuros).
Sem parametrização de tipo, cada novo adapter exigiria casts ou união
explícita que cresce com o tempo.

## Decisão

Tornar `BaseAgent` genérico em duas variáveis de tipo:

```python
class BaseAgent(ABC, Generic[Item, Output]):
    def run_batch(self, items: list[Item]) -> BatchResult[Output]: ...
    @abstractmethod
    def _process_single(self, item: Item) -> Output: ...
