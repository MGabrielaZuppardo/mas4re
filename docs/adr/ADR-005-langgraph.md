# ADR-005: Orquestração do pipeline via LangGraph StateGraph

- **Status:** Accepted
- **Data:** 2026-05-16
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2

## Contexto

O MAS4RE precisa comparar empiricamente duas arquiteturas de pipeline
LLM — multi-agente (classifier + prioritizer) versus agente único
(baseline) — sob condições controladas (SQ2). Essa comparação só é
válida se a orquestração for explícita, determinística e idêntica
entre execuções, permitindo isolar o efeito da decomposição
arquitetural das demais variáveis.

Até então, a orquestração multi-agente vivia em script imperativo
(`scripts/run_mas_pipeline.py`): sem nós nomeados, sem fronteiras
claras entre etapas, sem ponto único para instrumentar custo e falhas,
e difícil de diagramar ou citar no artigo. A estrutura `pipeline/`
existia apenas como stubs vazios.

Forças em jogo:
- **Validade experimental (SQ2):** o fluxo precisa ser reproduzível e
  comparável entre arquiteturas.
- **Instrumentação (SQ3, futura):** é preciso um ponto de extensão
  natural para o nó de cross-check (ADR-003).
- **Transparência/citabilidade:** o artigo exige uma arquitetura
  inspecionável e referenciável ("o grafo da Figura X").
- **Desacoplamento:** os agentes não devem conhecer o controle de
  fluxo nem a construção de LLM (ADR-002).

## Decisão

Implementar a orquestração do pipeline como `StateGraph` do LangGraph,
em duas fases:

**Fase 1 — grafo mínimo (este ADR / PR `feat/langgraph-minimal`):**
- Nós: `classifier_node`, `prioritizer_node`.
- Estado: `PipelineState` (Pydantic), reutilizado da camada de domínio
  como schema do grafo — fonte única de verdade, sem `TypedDict`
  paralelo.
- Arestas: START → classifier → prioritizer → END.
- Agentes injetados nos nós via `functools.partial` (ADR-002),
  mantendo o grafo desacoplado da construção de LLM.

**Fase 2 — extensões futuras (fora do escopo deste PR):**
- `cross_check_node` para detecção de conflito inter-agentes
  (ADR-003), habilitando a SQ3.
- `elicitor_node` como nó inicial (ADR-010).
- Arestas condicionais para tratamento de falhas (ex.: pular o
  prioritizer se o classifier produziu apenas falhas).

A futura `PipelineStrategy` (ADR-004) invocará
`compiled_graph.invoke(state)`, isolando a escolha de arquitetura
(MAS vs. baseline) no runner experimental.

## Consequências

**Positivas:**
- Fluxo = código legível e diagramável.
- Compatível com o ecossistema langchain/langgraph.
- Suporte nativo a checkpoints e replay.
- Reúso do `PipelineState` evita divergência entre domínio e pipeline.

**Negativas:**
- Dependência adicional (`langgraph`, já em `pyproject.toml`).
- Curva de aprendizado para quem não conhece grafos de estado.
- Acopla o pipeline à superfície de API do LangGraph.

**Neutras:**
- `pipeline/state.py` reexporta o tipo canônico de estado do grafo.

## Alternativas consideradas

- **Sequência imperativa:** estado anterior; descartada — sem
  visibilidade nem nós nomeados.
- **Celery/Airflow:** overkill para batch síncrono local.
- **Função composta (`prioritize(classify(state))`):** descartada —
  dificulta arestas condicionais e introspecção, e não escala para
  o grafo da Fase 2.

## Impacto na pesquisa

Na Fase 2, o `cross_check_node` permitirá à SQ3 observar conflitos
inter-agentes em nó dedicado, com instrumentação consistente. O
pipeline torna-se artefato citável: "o grafo da Figura X foi
compilado com LangGraph 1.1.6".

## Referências
- `pipeline/{graph,state}.py`, `pipeline/nodes/{classifier,prioritizer}_node.py`
- ADRs relacionados: ADR-002 (DI por construtor), ADR-003 (taxonomia
  de falhas / cross-check), ADR-004 (Strategy + Runner), ADR-010
  (Elicitor)