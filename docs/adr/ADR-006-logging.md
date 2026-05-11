# ADR-006: Logging estruturado, sem `print`

- **Status:** Accepted
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** infra

## Contexto

Agentes usam `logging.getLogger(__name__)` corretamente, mas
`scripts/run_baseline.py:134–173` e `evaluation/reporter.py:56–80`
intercalam `print()`. Sem `run_id` em logs, não é possível correlacionar
um requisito ao seu trace de LLM.

## Decisão

1. Banir `print` em todo o código (exceto teste).
2. `config/logging.py` configura dois modos:
   - dev: `rich.logging.RichHandler` (humano colorido).
   - prod/CI: `python-json-logger` (estruturado JSON).
3. `run_id` propagado via `contextvars` — todo log dentro de um run
   carrega o ID sem passar parâmetro.
4. CLI usa `rich.console.Console` para *output ao usuário* — separado
   de log.
5. Toda chamada de LLM mede `latency_ms` e grava em `trace.jsonl`
   (ADR-003).

## Consequências

**Positivas:**
- Correlação requisito↔chamada LLM↔erro via `run_id`.
- Logs ingestíveis por ELK/Loki sem parsing ad-hoc.
- Métrica de eficiência (latência) já está nos logs por padrão.

**Negativas:**
- Migrar `print`s existentes (commit dedicado na Semana 6).

**Neutras:**
- Acrescenta dependência `python-json-logger` (lightweight).

## Alternativas consideradas

- **`structlog`:** considerado; `logging` + JSON formatter cobre o caso
  com menos overhead conceitual.

## Impacto na pesquisa

Latência por chamada vira dado de eficiência (SQ2 — "pipeline é mais
lento que baseline?"). Trace correlacionado habilita análise de SQ3.

## Referências
- `config/logging.py` (a criar), `scripts/*.py`, `evaluation/reporter.py`
- ADRs relacionados: ADR-003