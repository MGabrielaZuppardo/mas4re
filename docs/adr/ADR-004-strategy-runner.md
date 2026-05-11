# ADR-004: Strategy Pattern + ExperimentRunner para validade de SQ2

- **Status:** Accepted
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2

## Contexto

`scripts/run_baseline.py` (224 ll.) e `scripts/run_mas_pipeline.py`
(229 ll.) são quase irmãos copy-paste, com IO, métrica e ordering
duplicados. Em SQ2 isso é grave: qualquer divergência entre os dois
scripts vira variável de confusão na comparação.

## Decisão

Aplicar Strategy Pattern:
- `OrchestrationStrategy` (ABC) — única coisa que diferencia baseline e
  pipeline (`execute(dataset, run_config) -> PipelineState`).
- `BaselineStrategy` e `PipelineStrategy`.
- `ExperimentRunner` único: carrega dataset → fixa seed → invoca strategy
  → coleta métricas → persiste `manifest.json` + `metrics.json` +
  `trace.jsonl` + `failures.jsonl`.

`manifest.json` registra `dataset_hash`, `seed`, `model`, `prompt_version`,
`detector_chain_version`, `git_commit`, `strategy` — chave para
comparabilidade.

## Consequências

**Positivas:**
- Validade da comparação SQ2 garantida por construção: mesmo caminho de
  IO, métrica e serialização.
- Comparar runs = `diff manifest_baseline.json manifest_pipeline.json`.
- Nova estratégia (ex.: hierarchical) entra como subclasse.

**Negativas:**
- Strategy ABC adiciona uma camada de indireção.

**Neutras:**
- Scripts antigos removidos; CLI Typer toma seu lugar.

## Alternativas consideradas

- **Manter scripts irmãos:** descartada — duplicação invalida SQ2.
- **Template Method em vez de Strategy:** descartada — o que varia é a
  orquestração inteira, não passos individuais.

## Impacto na pesquisa

Comparação SQ2 deixa de depender de disciplina manual ("lembrei de
atualizar os dois scripts?") e passa a ser garantida pela arquitetura.

## Referências
- `experiments/runner.py`, `experiments/strategies/*.py` (a criar)
- ADRs relacionados: ADR-005, ADR-003