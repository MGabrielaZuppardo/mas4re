# ADR-005: Orquestração do pipeline via LangGraph StateGraph

- **Status:** Accepted
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2

## Contexto

A estrutura `pipeline/` está vazia (todos `.py` 0 linhas), mas a
arquitetura pretendida (e documentada no plano) é LangGraph. Hoje a
orquestração mora em script imperativo (`scripts/run_mas_pipeline.py`)
sem visibilidade de fluxo, sem checkpoints, sem nós nomeados.

## Decisão

Implementar a orquestração do pipeline como `StateGraph` do LangGraph:
- Nós: `elicitor_node`, `classifier_node`, `prioritizer_node`,
  `cross_check_node`.
- Estado: `PipelineState` (Pydantic) propagado entre nós.
- Arestas: START → elicitor → classifier → prioritizer → cross_check → END.
- Arestas condicionais possíveis para tratamento de falhas (ex.: pular
  prioritizer se classifier produziu apenas falhas).

`PipelineStrategy` (ADR-004) invoca `compiled_graph.invoke(state)`.

## Consequências

**Positivas:**
- Fluxo = código legível e diagramável.
- Compatível com ferramentas do ecossistema langchain.
- Suporte nativo a checkpoints e replay.

**Negativas:**
- Dependência adicional (já listada em `pyproject.toml`).
- Curva de aprendizado para quem não conhece grafos de estado.

**Neutras:**
- `pipeline/state.py` define o tipo canônico de estado do grafo.

## Alternativas consideradas

- **Sequência imperativa:** atual; descartada — sem visibilidade.
- **Celery/Airflow:** overkill para batch síncrono.
- **Função composta (`prioritize(classify(elicit(state)))`):** descartada
  — dificulta arestas condicionais e introspecção.

## Impacto na pesquisa

Permite a SQ3 observar conflitos inter-agentes em nó dedicado
(`cross_check_node`) com instrumentação consistente. Pipeline torna-se
artefato citável: "o grafo da Figura X foi compilado com LangGraph
0.X.Y".

## Referências
- `pipeline/{graph,state}.py`, `pipeline/nodes/*.py`
- ADRs relacionados: ADR-004, ADR-003