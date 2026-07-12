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

## Nota de Atualização — 2026-07-12

O escopo real de `experiments/` cresceu além do descrito nesta ADR.

### Terceira estratégia: `TwoCallBaselineStrategy`

Adicionada a `experiments/strategy.py` junto com `BaselineStrategy` e
`PipelineStrategy`. É uma **ablação para a SQ3/RQ3**, não uma variante
de produção: usa `agents/two_call_baseline.py`, faz duas chamadas LLM
sequenciais com os mesmos templates de prompt de `ClassificationAgent`
+ `PrioritizationAgent`, mas passa o resultado intermediário via um
dataclass Python simples em vez do `PipelineState` validado por
Pydantic. Objetivo: isolar a contribuição do **contrato de estado
tipado** (`PipelineState`) da contribuição do desenho dos prompts —
ou seja, decompõe o ganho do `PipelineStrategy` em "dois-chamados" vs.
"dois-chamados-com-estado-tipado".

### Scripts de análise/ablation em `experiments/`

Não previstos nesta ADR, agora presentes:

| Script | Papel |
|---|---|
| `run_ablation_grid.py` | Grid de 6 condições: `TwoCallBaseline` vs `Pipeline`, isolando o contrato de estado tipado |
| `run_grid_full.py` | Grid completo n=625 (dataset PROMISE NFR+ completo), 12 condições sequenciais |
| `run_mistral_pipeline_fix.py` | Rerun seletivo (2 das 12 condições do grid) para validar fix de parser no `mistral:7b` |
| `run_classifier.py` | Roda `ClassificationAgent` isolado no PROMISE NFR+, por modelo configurado |
| `compute_stats.py` | Cálculo estatístico principal (RQ1: Wilcoxon signed-rank + Cohen's h pareado por requisito) |
| `compute_ablation_stats.py` | Estatística da ablação `TwoCallBaseline` vs `Pipeline`, mesmo protocolo da RQ1 |
| `compute_ablation_bootstrap.py` | IC bootstrap do delta F1-macro (pipeline − two_call_baseline) |
| `compute_rq3_kappa_bootstrap.py` | IC bootstrap do incremento de kappa (Fleiss', concordância MoSCoW entre modelos) na RQ3 |
| `compute_baseline_vs_two_call_stats.py` | Comparação ad-hoc baseline vs two_call_baseline (não reportada diretamente no artigo) |
| `compute_subcategory_errorprop.py` | Métricas de subcategoria NFR (10 classes) + propagação de erro de confidence < 0.70 no pipeline |
| `cohens_h.py` | Utilitário: cálculo de Cohen's h |
| `verify_kappa_and_parsing.py` | Verificação ad-hoc: kappa leave-one-model-out + parsing, fora do pipeline do artigo |
| `test_parser_fix.py` | Smoke test do fix de schema em `ClassificationAgent._parse_response` |

### CLI

A previsão "Scripts imperativos serão substituídos pelo CLI (PR
seguinte)" **se concretizou**: `cli/main.py` (Typer, comandos
`run`/`eval`/`compare`) já usa `ExperimentRunner` +
`BaselineStrategy`/`PipelineStrategy` diretamente. `TwoCallBaselineStrategy`
ainda não está exposta pela CLI — permanece acessível só via os scripts
de `experiments/` listados acima.

Decisão original (Strategy + ExperimentRunner) permanece válida; a ADR
estava desatualizada apenas quanto ao inventário de estratégias e
scripts, agora coberto acima. Recomenda-se uma ADR de sequência formal
para `TwoCallBaselineStrategy` caso ela seja promovida a comparação
reportada no artigo (hoje é instrumentação interna da RQ3).
