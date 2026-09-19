# ADR-011: Capacidades agentic em ClassificationAgent/PrioritizationAgent

- **Status:** Accepted
- **Data:** 2026-09-14
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2 (contribuição atual: eixo único de coordenação, agora entre agentes de verdade)

## Contexto

Uma auditoria de código nesta sessão confirmou que `ClassificationAgent`/`PrioritizationAgent`
eram, até aqui, chamadas único-tiro ao LLM: monta prompt → `self._llm.invoke(messages)` uma vez →
parse → retorna. Sem memória entre chamadas, sem loop de raciocínio, sem ferramentas, sem
autonomia sobre a própria ação. Pela literatura clássica (Wooldridge & Jennings, 1995 —
autonomia, reatividade, pró-atividade, capacidade social) e pelos frameworks modernos de LLM
agents (Xi et al., 2023; Sumers et al., 2023 — CoALA), isso não se qualifica como "agente" no
sentido forte — o que enfraquecia a base inteira do MAS4RE, não só um detalhe isolado, já que a
dissertação em andamento (ADR-009, Condição A vs. B) compara estratégias de coordenação *entre*
esses agentes.

Uma alternativa considerada foi tratar "capacidade do agente" como um segundo eixo experimental
independente (classes agentic paralelas, ao lado das existentes). Rejeitada: a dissertação
mantém um eixo de pesquisa só (estratégia de coordenação, A vs. B) — as capacidades agentic
entram dentro das mesmas classes que a Condição A/B já usam, não como uma segunda variável.

## Decisão

Modificar `ClassificationAgent`/`PrioritizationAgent` in place, adicionando três componentes,
cada um fundamentado na literatura de LLM agents:

1. **Memória episódica** (`agents/memory.py::EpisodicMemory`) — CoALA (Sumers et al., 2023) /
   Reflexion (Shinn et al., 2023). Guarda os itens já processados no lote atual e recupera os *k*
   mais similares (similaridade lexical Jaccard, não embeddings — mantém determinismo sem
   dependência nova) como few-shot dinâmico para o próximo item.
2. **Ferramenta de grounding** (`agents/tools.py::make_nfr_taxonomy_tool`) — ReAct (Yao et al.,
   2023). Só o classifier usa: `lookup_nfr_taxonomy(category_code)` devolve a definição canônica
   de uma categoria NFR a partir da mesma taxonomia já injetada no prompt
   (`prompts/v1/_shared.py::build_nfr_block`/`default_nfr_categories`). O modelo decide se chama,
   bounded a no máx. 1 chamada por item (a geração de fechamento usa o LLM sem tools vinculadas).
3. **Autocrítica / planejamento com feedback** (`_critique` em cada agente) — Self-Refine
   (Madaan et al., 2023). Uma rodada: gerar → criticar → revisar se `needs_revision`. Qualquer
   falha na crítica (parse, schema inválido) degrada para a resposta original já validada, em vez
   de derrubar o item.

Critério de aceite (Wooldridge & Jennings, 1995): o texto da dissertação deve argumentar,
componente a componente, por que esta versão satisfaz autonomia (o modelo decide se chama a
ferramenta e se revisa) e pró-atividade (revisão dirigida a um objetivo de qualidade
autoavaliado) — o que a versão single-shot não satisfazia.

### Memória exige processamento em duas fases

Memória que cresce durante o lote cria uma dependência causal real (item N depende do que os
itens 1..N-1 já produziram) — estrutural a esse tipo de memória, não uma exigência de
determinismo que dê pra relaxar (`temperature=0.0`/`seed=42` do CLAUDE.md garantem determinismo
por chamada individual, não impõem processamento sequencial por si só).

Solução: **warm-up sequencial + memória congelada + resto em paralelo.**
`agents/base.py::BaseAgent._run_batch` processa os primeiros `_WARMUP_SIZE = 10` itens em
sequência (memória acumulando normalmente), chama `memory.freeze()`, e processa o restante via
`ThreadPoolExecutor` como antes — leitura concorrente de memória já congelada é segura (sem
escrita concorrente). Alternativas descartadas: lote inteiro sequencial (custo de runtime
desnecessário para n grande); memória estática pré-construída fora do run (perde o caráter de
aprendizado dentro da própria execução que dá à memória seu valor episódico).

Agentes sem `self._memory` setado (`BaselineAgent`/`TwoCallBaselineAgent`, e
`ClassificationAgent`/`PrioritizationAgent` quando chamados via
`pipeline/nodes/process_item_node.py` — o caminho do `pipeline_streaming`, que despacha por
`Send` e não por `_run_batch`) continuam 100% paralelos, sem warm-up — `self._memory` só é
setado dentro de `classify_batch()`/`prioritize_batch()`.

## Consequências

**Positivas:**
- Condição A (`build_pipeline_graph`) e Condição B (`build_mediated_pipeline_graph`) passam a
  comparar coordenação entre agentes que de fato se qualificam como agentes pela literatura, sem
  abrir uma segunda variável experimental.
- Ferramenta de grounding dá dado concreto para comparar taxa de `CATEGORY_HALLUCINATED`
  (`evaluation/failure_detectors.py::detect_category_hallucination`) antes/depois.
- Nenhuma mudança em `domain/models.py`/`domain/enums.py` — `ClassificationOutput`/
  `PrioritizationOutput` continuam os mesmos, `_critique` só produz instâncias validadas pelo
  mesmo schema.

**Negativas / trade-offs aceitos:**
- Custo de inferência maior por item: sempre +1 chamada de autocrítica, ocasionalmente +1 de
  tool-call. `pipeline_streaming` não ganha memória (limitação conhecida, documentada, não bug).
- Itens processados depois do warm-up só se beneficiam de memória dos primeiros ~10 itens do
  lote, não de tudo que já foi processado até ali.
- Paralelismo perdido só durante o warm-up (~10 itens), não no lote inteiro.

**Neutras:**
- `BaselineAgent`/`TwoCallBaselineAgent` (ablações de SQ2) não são tocados — continuam o
  contraponto deliberadamente mais simples que já eram.

## Alternativas consideradas

- **Classes agentic paralelas** (`AgenticClassificationAgent` etc., ao lado das existentes) —
  descartada: o usuário decidiu manter um eixo de pesquisa só, não dois.
- **Lote inteiro sequencial** — descartada: custo de runtime desnecessário (n=625 ficaria vários
  múltiplos mais lento) quando só a memória precisa de ordem, não o tool-use nem a autocrítica.
- **Memória estática pré-construída fora do run** — descartada: perderia o caráter de
  aprendizado dentro da própria execução (episódico) que a literatura (CoALA/Reflexion)
  distingue de memória semântica/estática.

## Impacto na pesquisa

Fortalece a validade de construto da comparação A vs. B: antes, "coordenação entre agentes"
comparava estratégias entre componentes que não se qualificavam como agentes; agora a mesma
comparação é feita entre agentes com memória, planejamento e ação, tal como definidos na
literatura de LLM agents citada acima.

## Referências

- Arquivos afetados: `agents/base.py`, `agents/classifier.py`, `agents/prioritizer.py`,
  `agents/memory.py` (novo), `agents/tools.py` (novo), `prompts/v1/classification.py`,
  `prompts/v1/prioritization.py`, `prompts/v1/_shared.py`
- Testes: `tests/unit/test_memory.py`, `tests/unit/test_tools.py`, extensões em
  `tests/unit/test_classifier.py`/`test_prioritizer.py`
- ADRs relacionados: ADR-002 (DI por construtor), ADR-003 (taxonomia de falhas), ADR-009
  (estratégia de coordenação condicional — a comparação que esta ADR fortalece)
- Literatura: Wooldridge & Jennings (1995); Xi et al. (2023); Sumers, Yao, Narasimhan & Griffiths
  (2023, CoALA); Yao et al. (2023, ReAct); Madaan et al. (2023, Self-Refine); Shinn et al.
  (2023, Reflexion)
