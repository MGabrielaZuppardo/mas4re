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
