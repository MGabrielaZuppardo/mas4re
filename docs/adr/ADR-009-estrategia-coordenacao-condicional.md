# ADR-009: Estratégia de coordenação condicional (Condição B)

- **Status:** Accepted
- **Data:** 2026-09-14
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2 (contribuição atual da dissertação: efeito de estratégias de coordenação)

## Contexto

A dissertação em andamento usa o MAS4RE exclusivamente como plataforma experimental para
responder: como diferentes estratégias de coordenação centralizada entre os agentes classificador
e priorizador influenciam o desempenho de classificação e priorização? Isso exige duas condições
experimentais comparáveis, variando *apenas* a estratégia de coordenação (desenho de fator único):

- **Condição A** (já existente, `pipeline/graph.py::build_pipeline_graph`): fluxo sequencial fixo
  `classifier → prioritizer → cross_check → END`, sem retroalimentação.
- **Condição B** (esta ADR): roteamento condicional mediado — quando `cross_check_node` detecta um
  conflito inter-agentes (`evaluation/cross_agent_check.py::detect_inter_agent_conflict`, já
  existente desde ADR-003: categoria NFR crítica classificada, mas rebaixada para
  Could/Won't-Have pelo priorizador), o lote é reenviado ao classificador para uma segunda
  passagem.

Até a implementação desta ADR, a Condição B existia apenas como descrição textual — nenhum dos
símbolos que a implementam (`build_mediated_pipeline_graph`, `MediatedPipelineStrategy`,
`PipelineState.retry_counts`, `finalize_mediation`) estava de fato escrito no repositório. Sem
essa condição executável, a pergunta de pesquisa central não tem como ser respondida com dados
reais.

## Decisão

Implementar a Condição B como retry **em nível de batch** (o lote inteiro volta ao classifier, não
apenas os itens em conflito) via `add_conditional_edges` do LangGraph, limitado a no máximo 2
passagens (1 retry).

Detalhamento:
- `pipeline/nodes/cross_check_node.py` é reusado, sem alterar seu comportamento observável para a
  Condição A: continua gravando `state.metrics["inter_agent_conflicts"]` via
  `check_requirements`. Passa também a incrementar `state.retry_counts["_pass"]` e a acumular um
  snapshot por passagem em `state.metrics["mediation_passes"]`. Por ser o mesmo nó nas duas
  condições, esses campos ficam inertes na Condição A (nada no grafo fixo os lê), preservando o
  fator único do experimento.
- A contagem de passagem é escrita dentro de `cross_check_node` — um nó real — nunca dentro da
  função de roteamento (`_route_after_cross_check`): o LangGraph só persiste mutações de estado
  devolvidas por nós, uma função de aresta condicional apenas decide o próximo nó.
- `finalize_mediation_node` é o nó terminal da Condição B: compara o conjunto de conflitos da
  primeira passagem com o da última para computar quantos foram resolvidos pelo retry, e grava o
  resumo em `state.metrics["mediation"]`.
- `experiments/strategy.py::MediatedPipelineStrategy` espelha `PipelineStrategy`, trocando apenas o
  grafo usado — mesmos agentes, mesmos modelos, mesma temperatura.

## Consequências

**Positivas:**
- A Condição B passa a ser executável de verdade, permitindo comparação empírica A vs. B sobre os
  mesmos datasets (PROMISE/NFRIC).
- `cross_check_node` compartilhado entre as duas condições garante que a única variável manipulada
  seja a topologia do grafo (presença ou ausência da aresta condicional), não uma reimplementação
  paralela do detector de conflito.
- `state.metrics["mediation"]` fornece, por execução, dado suficiente para RQs sobre taxa de
  resolução de conflito pós-retry, sem precisar reprocessar `results.json` externamente.

**Negativas / trade-offs aceitos:**
- Retry em nível de batch reprocessa itens que não estavam em conflito (custo de inferência
  duplicado para o lote inteiro, não só para os itens conflitantes) — trade-off aceito porque
  mantém o grão de comparação idêntico ao da Condição A (mesma unidade de execução: o lote), e
  porque um retry per-item quebraria a comparabilidade com o pipeline batch existente.
- `failure_detections` acumula os registros das duas passagens (a segunda chamada não substitui a
  primeira) — decisão deliberada: ambas as passagens são eventos reais que aconteceram, não ruído
  a descartar.

**Neutras:**
- Único campo novo no domínio (`PipelineState.retry_counts`); o histórico por passagem vive dentro
  de `metrics`, seguindo a mesma convenção que ADR-003 já adotou para `inter_agent_conflicts`.

## Alternativas consideradas

- **Retry per-item via `Send`** (mesmo mecanismo do pipeline streaming, ADR-005/
  `pipeline/graph_streaming.py`): reenviaria só os itens em conflito. Descartada porque muda o grão
  de comparação em relação à Condição A (que processa em lote) — isolar o efeito da *estratégia de
  coordenação* exige manter tudo mais constante possível, incluindo a granularidade do
  reprocessamento.
- **Contador de passagem mantido fora do estado (variável local no builder do grafo):**
  descartada — LangGraph reconstrói/reexecuta a partir do estado a cada invocação de nó; qualquer
  contador que precise sobreviver entre passagens do grafo precisa estar no `PipelineState`.
- **Máx. passagens configurável via `settings`:** descartada por ora — fixar em 2 (1 retry) mantém
  o desenho experimental simples e replicável; parametrizar é um ajuste trivial futuro se a análise
  pedir.

## Impacto na pesquisa

Sem esta ADR, a comparação Condição A vs. B (o eixo central da dissertação atual) não tinha
nenhuma implementação a comparar. Com ela, é possível rodar a mesma grade de experimentos
(`scripts/run_grid.py` e afins) trocando apenas a estratégia (`pipeline` vs. `pipeline_mediated`)
e obter `results.json` comparáveis, com `metrics.mediation` como dado bruto para as RQs sobre
efeito do roteamento condicional e taxa de resolução de conflito.

## Referências

- Arquivos afetados: `pipeline/graph.py`, `pipeline/nodes/cross_check_node.py`,
  `experiments/strategy.py`, `experiments/runner.py`, `cli/main.py`, `domain/models.py`
- Testes: `tests/unit/test_graph_mediated.py`, `tests/unit/test_strategy_mediated.py`
- ADRs relacionados: ADR-003 (taxonomia de falhas / cross-check), ADR-004 (Strategy + Runner),
  ADR-005 (LangGraph — nota de atualização de 2026-07-18 sobre `cross_check_node`)
