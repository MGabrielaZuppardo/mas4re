# ADR-012: Agentes autônomos (laço ReAct) com mensagens tipadas

- **Status:** Accepted
- **Data:** 2026-09-20
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2 (a coordenação passa a ser entre agentes que decidem por conta própria)

## Contexto

O ADR-011 acrescentou memória, ferramenta e autocrítica aos agentes, mas o **código** continuou
decidindo a sequência: a recuperação de memória era injetada sempre no prompt, a ferramenta tinha no
máximo uma chamada e a resposta final saía de um LLM sem ferramentas, a crítica disparava por
`_needs_critique` (regra sobre confiança), a crítica do priorizador era incondicional e o
`priority_rank` era pedido ao LLM e sobrescrito pelo código. Na definição de Anthropic (2024),
*workflows* são "sistemas em que LLMs e ferramentas são orquestrados por caminhos de código
predefinidos", e *agentes* são sistemas em que "o LLM dirige dinamicamente o próprio processo e o uso
de ferramentas". Pelo critério deles, e pelo de Wooldridge & Jennings (1995: autonomia, reatividade,
pró-atividade, capacidade social), o que existia era um workflow com LLM dentro. Como o objeto de
pesquisa é a coordenação **entre agentes**, isso ameaçava a validade de construto do experimento.

## Decisão

Criar dois agentes em `agents/agentic/` cujo trabalho em cada item é um **episódio de um laço
observar → decidir → agir** (ReAct, Yao et al., 2023; ciclo percepção-ação de Russell & Norvig): o
modelo escolhe qual ferramenta chamar, se chama alguma, quantas vezes, e quando entregar a resposta.
O código só fornece o espaço de ações, valida cada ação contra o esquema (ação inválida vira
observação de erro que o agente pode corrigir e é contada) e impõe um orçamento de passos. Estourar o
orçamento faz o item **falhar** (vai para `failed.csv`), sem rótulo inventado.

- **Classificador** — ações: `lookup_nfr_taxonomy`, `recall_similar_cases` (memória episódica,
  CoALA), `check_consistency` (verificação externa determinística, CRITIC: Gou et al., 2024, em vez
  de confiar na autoconfiança do modelo), `submit_classification` (encerra) e, ao receber um pedido
  do par, `reply_to_request` (aceitar revendo a classificação ou recusar).
- **Priorizador** — ações: `recall_similar_cases`, `backlog_overview` (composição do lote e
  prioridades já decididas), `request_reclassification` (pede ao classificador que reconsidere),
  `submit_priority` (encerra).
- **Mensagens tipadas** (estilo FIPA-ACL, `domain/messages.py`): `request`, `inform`, `agree`,
  `refuse`, `failure`, com conteúdo tipado e `in_reply_to`. O priorizador só conhece o trabalho do
  classificador pela mensagem `inform` que este publica (inclui a confiança, antes descartada).
- **Objetivo e crenças explícitos** (Rao & Georgeff, 1995): cada episódio registra o `Goal` e as
  `Belief` (com a fonte: percepção, ferramenta ou mensagem), e ambos aparecem no prompt.
- **Registro de ações** (`domain/agent_actions.py`, `actions.jsonl`): ação, argumentos, observação,
  tokens e latência por passo, no formato de dado de treino.

O prompt descreve papel, objetivo, definições, ferramentas e como o trabalho termina, **nunca um
procedimento**. Os agentes do ADR-011 (`ClassificationAgent`/`PrioritizationAgent`) permanecem
intactos como a condição `workflow` do controle da ablação (`pipeline` × `pipeline_agentic`).

## Consequências

**Positivas:**
- A comparação A × B passa a ser entre agentes que se qualificam como tais; os sete critérios de
  conformidade (`docs/agency_conformance.md`) são testáveis, cinco deles em CI com LLM roteirizado.
- Some o rótulo fabricado por falha de parse ("F", confiança 0.0): itens sem resposta viram falhas
  visíveis, com cobertura reportada ao lado das métricas.
- O `actions.jsonl` e o bloco `agent_activity` (ações válidas, término por orçamento, entropia das
  sequências, tokens) medem se o modelo de fato age como agente.

**Negativas / trade-offs aceitos:**
- Mais chamadas de LLM por item que o workflow. Modelos de 7–8B podem ter baixa taxa de ações bem
  formadas; não há rede de segurança em forma de workflow (seria voltar ao problema). A taxa entra no
  relatório e alimenta a decisão de treinar modelos maiores.
- O código dos agentes duplica pouco (`_detection_context`) em vez de tocar o workflow.
- A capacidade social existe nos agentes mas **não** está ligada às condições A/B; o protocolo de
  pedidos (P2) e o mediado (P3) ficam para a `feat/protocol-conditions`.

**Neutras:**
- `PipelineState.messages` é a caixa de mensagens do ambiente. `pipeline/graph.py` e os nós passaram a
  tipar o agente como `BaseAgent`, sem mudança de comportamento.

## Alternativas consideradas

- **`create_react_agent` do LangGraph** — descartada: o registro por passo, os tokens e o teste de
  topologia exigem controle fino, e a API prebuilt está migrando para `langchain.agents.create_agent`.
- **Reescrever os agentes do ADR-011 no lugar** — descartada: sem o workflow não há controle para a
  ablação (C7).
- **Rede de segurança em workflow quando o modelo não chama ferramentas** — descartada: esconde
  justamente o que se quer medir.

## Impacto na pesquisa

Fortalece a validade de construto: "coordenação entre agentes" deixa de comparar componentes que só
preenchem um campo JSON. O regime de informação continua `zero_shot` (a memória guarda só respostas do
próprio agente, nunca gabarito). Os números de C5 (a observação muda a ação) e C7 (ablação contra o
workflow) saem de execuções reais, não de teste unitário.

## Referências

- Arquivos afetados: `agents/agentic/`, `domain/agent_actions.py`, `domain/messages.py`,
  `evaluation/agent_activity.py`, `prompts/v2/`, `experiments/strategy.py`, `experiments/runner.py`,
  `cli/main.py`, `pipeline/graph.py`
- Testes: `tests/unit/test_agentic_*.py`, `test_agency_conformance.py`, `test_agent_activity.py`,
  `test_messages.py`
- ADRs relacionados: ADR-009 (coordenação), ADR-011 (agentes com memória e ferramentas, mantidos como
  controle)
- Literatura: Wooldridge & Jennings (1995); Yao et al. (2023, ReAct); Sumers et al. (2023, CoALA);
  Gou et al. (2024, CRITIC, ICLR); Rao & Georgeff (1995, BDI, ICMAS); FIPA ACL Message Structure
  Specification (SC00061); Anthropic (2024), *Building effective agents*
