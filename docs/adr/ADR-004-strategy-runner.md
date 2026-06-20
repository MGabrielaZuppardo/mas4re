# ADR-004: Strategy Pattern + ExperimentRunner para validade da SQ2

- **Status:** Accepted
- **Data:** 2026-05-16
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2

## Contexto

A SQ2 compara empiricamente duas arquiteturas (multi-agente vs.
agente único). A comparação só é válida se ambas forem executadas
sob condições idênticas — mesmo dataset, mesma seed, mesmo runner —
variando apenas a arquitetura. A orquestração vivia em scripts
imperativos distintos (`run_baseline.py`, `run_mas_pipeline.py`),
com caminhos de código divergentes e sem manifesto reprodutível.

## Decisão

Introduzir o **Strategy Pattern**:
- `OrchestrationStrategy` (ABC): contrato
  `execute(requirements) -> PipelineState`.
- `BaselineStrategy`: agente único (1 chamada LLM).
- `PipelineStrategy`: grafo LangGraph (classifier → prioritizer,
  ADR-005).

E o **`ExperimentRunner`**: carrega dataset, executa a estratégia
sob `RunConfig` congelado e grava `manifest.json` com `git_commit`,
`seed`, `dataset_md5`, `model`, `strategy`, `langgraph_version` e
`timestamp`.

## Implementação

- `experiments/strategy.py`: ABC + `BaselineStrategy` +
  `PipelineStrategy` + `_coerce_state` (normaliza retorno do
  LangGraph dict→PipelineState).
- `experiments/runner.py`: `RunConfig`, `RunResult`,
  `ExperimentRunner.execute(strategy, config)` + manifesto.
- `tests/unit/test_experiment_runner.py`: runner, manifesto,
  determinismo via `FakeStrategy`.
- O runner também computa métricas (classificação via gold labels
  do PROMISE; subcategoria; distribuição MoSCoW) e grava
  `results.json` (predições + métricas) ao lado de `manifest.json`.
  Priorização não tem gold label no PROMISE → apenas distribuição
  (ameaça discutida em §7 do artigo).

## Consequências

**Positivas:**
- Variável arquitetural isolada → comparação SQ2 controlada.
- `manifest.json` torna cada run reprodutível e citável
  (critério Verificabilidade/Transparência do SBCARS).
- Scripts imperativos serão substituídos pelo CLI (PR seguinte).

**Negativas:**
- Indireção extra (Strategy) para quem lê o código pela 1ª vez.

**Neutras:**
- `RunConfig` centraliza os parâmetros do experimento.

## Alternativas consideradas

- **Dois scripts independentes:** estado anterior; descartado —
  caminhos divergentes invalidam a comparação.
- **Flag condicional dentro de um script:** descartada — mistura
  responsabilidades e dificulta teste isolado.

## Impacto na pesquisa

Habilita a execução do grid da SQ2 (3 modelos × 2 idiomas × 2
arquiteturas) sob protocolo único, com manifesto por run para
auditoria e replicação independentes.

## Referências
- `experiments/{strategy,runner}.py`,
  `tests/unit/test_experiment_runner.py`
- ADRs relacionados: ADR-005 (LangGraph), ADR-002 (DI), ADR-003
  (taxonomia de falhas — instrumentação futura no runner)
