# ADR-003: Taxonomia de falhas e detectores plugáveis para SQ3

- **Status:** Accepted
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ3

## Contexto

SQ3 investiga **padrões de falha** (ambiguidade, alucinação, conflito
inter-agentes). O código atual usa `except Exception → fallback object
com confidence=0.0`, contaminando métricas: falha de infra vira
"predição errada". Sem categorização, é impossível responder
"o pipeline alucina mais que o baseline?".

## Decisão

Introduzir:
1. `FailureMode` (`StrEnum`): INFRA_TRANSIENT, SCHEMA_INVALID,
   CATEGORY_HALLUCINATED, LOW_CONFIDENCE, GROUNDING_MISSING,
   AMBIGUOUS, INTER_AGENT_CONFLICT.
2. `FailureRecord` (Pydantic): req_id, stage, mode, severity
   (`fatal | degraded | flagged`), evidence, timestamp.
3. `DetectorChain` versionada, plugável, executada após cada parse.
4. `BatchResult[T]`: `successes`, `failures`, `flagged` separados.
5. `cross_agent_check`: detector de conflito inter-agentes (só pipeline).

Métricas separam `quality` (sobre successes∪flagged) e `failure` (taxa
por modo).

## Consequências

**Positivas:**
- Falha categorizada é dado científico para SQ3, não ruído.
- Comparação baseline×pipeline ganha dimensão de "tipos de falha".
- Reanálise offline possível via `trace.jsonl` sem rerun.

**Negativas:**
- Cada agente roda a cadeia de detectores (custo computacional baixo).
- Manutenção da taxonomia conforme novos modos forem observados.

**Neutras:**
- DetectorChain versionada: muda versão = muda análise sem perder histórico.

## Alternativas consideradas

- **Manter except Exception genérico:** descartada — invalida SQ3.
- **LLM-as-judge único:** considerado como detector v2 futuro; v1 usa
  heurísticas determinísticas para baseline.

## Impacto na pesquisa

Resposta direta para SQ3. INTER_AGENT_CONFLICT só existe no pipeline,
gerando achado próprio: o trade-off de adicionar coordenação.

## Referências
- `domain/failures.py` (a criar), `evaluation/failure_detectors.py`,
  `evaluation/cross_agent_check.py`, `evaluation/trace.py`
- docs/sq3_methodology.md

## Nota de Atualização — 2026-07-18

Até esta data, `DetectorChain`/`default_chain` e `cross_agent_check` estavam
implementados e testados, mas **nunca eram chamados** por nenhum agente, node
ou runner — só existiam nos próprios testes unitários. Isso foi corrigido:

- `evaluation/cross_agent_check.py::check_requirements` agora roda dentro de
  `pipeline/nodes/cross_check_node.py`, que está plugado em
  `pipeline/graph.py` (`classifier → prioritizer → cross_check → END`).
  `experiments/runner.py::_compute_metrics` propaga
  `state.metrics["inter_agent_conflicts"]` para `results.json`.
- `evaluation/failure_detectors.py::DetectorChain` (via `default_chain()`)
  agora roda por item dentro de `agents/base.py::BaseAgent._run_batch`
  (novo hook `_detection_context`, implementado em `ClassificationAgent`,
  `PrioritizationAgent`, `BaselineAgent` e `TwoCallBaselineAgent` — este
  último replica a mesma lógica dentro do seu próprio loop com `tqdm`, já
  que não usa o `_run_batch` genérico). Os registros são persistidos em
  `state.metrics["failure_detections"]` e propagados para `results.json`
  do mesmo jeito que `inter_agent_conflicts`.
- Ressalva conhecida: `normalize_nfr_category` (`domain/nfr_normalization.py`)
  descarta categorias NFR não reconhecidas para `None` **antes** do
  `DetectorChain` rodar, então `detect_category_hallucination` raramente
  dispara para as stages `classify`/`baseline` — a alucinação já foi
  filtrada rio acima, não é capturada aqui. Capturar a categoria bruta
  pré-normalização para alimentar esse detector de forma mais sensível
  ainda é trabalho futuro, não feito.
- `BatchResult[T]` (item 4 da decisão original) segue não utilizado — os
  registros de falha são acumulados como `list[FailureRecord]` simples, sem
  a partição successes/failures/flagged que essa classe modelaria.
- **Atualização same-day:** a assimetria entre "JSON mal formado" e "violação
  de constraint do Pydantic" (ambos caindo no mesmo `except Exception`
  genérico em `classifier.py`/`baseline.py`/`prioritizer.py`, virando o mesmo
  fallback sem nunca acionar `SCHEMA_INVALID`) **foi corrigida**. Agora só a
  decodificação do JSON fica no fallback seguro (sem retry); um
  `RequirementType`/`MoSCoWPriority` inválido ou um `pydantic.ValidationError`
  (ex.: `confidence=1.4`) propaga por `_process_single` → `@llm_retry` → se
  persistir após 3 tentativas, o item cai fora do batch e
  `detect_schema` finalmente dispara `SCHEMA_INVALID` de verdade — unificando
  o comportamento com o que `two_call_baseline.py` já fazia por acidente
  (ver `agents/two_call_baseline.py`, onde a validação Pydantic do
  `BaselineOutput` final nunca esteve dentro de um try/except).
