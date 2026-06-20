# ADR-010: Elicitor anota o requisito, não modifica seu texto

- **Status:** Proposed
- **Data:** 2026-05-10
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ1, SQ3

## Contexto

O agente Elicitor (Semana 6) detecta ambiguidades e candidatos a
refinamento. Há duas vertentes possíveis:
1. **Modificar** o texto do requisito (reescrita assistida).
2. **Anotar** o requisito com flags/observações, deixando o texto
   original intacto.

A escolha afeta diretamente o que SQ1 (eficácia) e SQ3 (ambiguidade
como modo de falha) medem.

## Decisão (proposta — a confirmar com orientador)

Elicitor **anota** o requisito com:
- `ambiguity_score: float`
- `ambiguity_reasons: list[str]`
- `suggested_clarifications: list[str]` (opcional, geradas mas não
  aplicadas)

O texto original do requisito **não é alterado**. Classifier e
Prioritizer recebem o requisito original mais anotações como contexto
adicional do estado.

## Consequências

**Positivas:**
- Comparação SQ2 fica honesta: baseline e pipeline recebem o mesmo
  texto de entrada.
- "Ambiguidade" vira `FailureMode.AMBIGUOUS` mensurável (SQ3).
- Reversível: se a banca quiser, comparar com modo "modifica" depois.

**Negativas:**
- Pipeline não "consome" automaticamente a sugestão do Elicitor.

**Neutras:**
- Sugestões ficam disponíveis no `trace.jsonl` para análise qualitativa.

## Alternativas consideradas

- **Modificar texto:** descartada por ora — introduz variável-de-confusão
  forte no baseline vs pipeline (não compararíamos a mesma entrada).
- **Elicitor com aprovação humano-in-the-loop:** fora de escopo da
  dissertação atual.

## Impacto na pesquisa

Define como ambiguidade entra na taxonomia de falhas SQ3 e mantém a
comparação SQ2 com entradas idênticas.

## A confirmar
- Validar com orientador antes de mover status para `Accepted`.

## Referências
- `agents/elicitor.py`, `prompts/v1/elicitation.py`
- ADRs relacionados: ADR-003, ADR-005
