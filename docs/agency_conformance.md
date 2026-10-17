# Agency conformance

Um componente só é chamado de *agente* neste projeto se passar nestes critérios. Se um falhar, o
texto usa "fluxo de trabalho multiestágio" para aquele componente. A regra que separa os dois é uma
pergunta: **quem decide o próximo passo?** Se é o código, é workflow; se é o modelo, é agente
(Anthropic, 2024; Wooldridge & Jennings, 1995).

| # | Critério | Como se verifica | Falha = ainda é workflow |
|---|---|---|---|
| C1 | O próximo passo é ação emitida pelo LLM, não aresta de código | Topologia do grafo sem nós de passos internos; laço sem nomes de ferramentas; agentes nunca chamam ferramentas; LLM roteirizado com ações diferentes muda o caminho; entropia das sequências de ações > 0 em execuções reais | todos os itens seguem a mesma sequência |
| C2 | Quem decide terminar é o agente; o código só impõe orçamento | `submit_*` é ação do agente; estouro de orçamento faz o item falhar, sem resposta inventada; taxa de término por orçamento reportada | quase tudo termina por orçamento |
| C3 | Os agentes só se comunicam por mensagens tipadas | Teste estático (o priorizador não lê `classified_requirements` nem importa o classificador) e teste de comportamento (estado contraditório é ignorado) | leitura direta do estado do par |
| C4 | O agente pode recusar um pedido | Aceite e recusa possíveis, decididos só pelo modelo; taxas de aceite e recusa, ambas > 0, em execuções reais | aceita sempre: o "conselho" é ordem disfarçada |
| C5 | Uma observação altera a ação seguinte | Unitário: a observação chega ao modelo e a resposta a ela muda a ação. Em execuções reais: associação estatística entre resultado da ferramenta e ação escolhida | a ação não depende do que foi observado |
| C6 | Objetivos e crenças explícitos e inspecionáveis | `Goal` e `Belief` no episódio (`actions.jsonl`) e presença comprovada no prompt | não há onde inspecionar |
| C7 | Ablação contra o workflow nos mesmos itens | `pipeline` (workflow) × `pipeline_agentic`, com custo, mesmo que o ganho seja nulo | a comparação não existe |

## Estado

| # | CI (LLM roteirizado) | Execução real |
|---|---|---|
| C1 | `test_agency_conformance.py::TestC1…` | `agent_activity.action_sequence_entropy_bits` no `results.json` |
| C2 | `TestC2…` | `agent_activity.budget_exhaustion_rate` |
| C3 | `TestC3…` | n/a (propriedade do código) |
| C4 | `TestC4…` | taxas de `agree`/`refuse` — só existirão com o protocolo P2 (`feat/protocol-conditions`) |
| C5 | `test_agentic_runtime.py::TestTheModelDecides` | `analyze_agency.py` (a fazer) |
| C6 | `TestC6…` | inspeção de `actions.jsonl` |
| C7 | n/a | comparação `pipeline` × `pipeline_agentic` (a fazer, na Rodada) |

## O que o código decide e o que o modelo decide

| Decidido pelo código (é o ambiente) | Decidido pelo LLM (é o agente) |
|---|---|
| Quais itens chegam, o conjunto de ferramentas, o orçamento de passos, a validação de esquema, de quem é a vez | Qual ferramenta chamar, se chama alguma, em que ordem, quantas vezes, quando concluir, o que dizer ao par, se aceita ou recusa um pedido |

Proibido no código dos agentes: `if confiança < x: verificar`, chamadas fixas de ferramenta e um
procedimento ("consulte a taxonomia e depois revise") no prompt. A política de retenção da memória
(só respostas confiantes e consistentes viram exemplo) é higiene de memória do ambiente, não controle
de fluxo do agente.

## Limites conhecidos

- A capacidade social (pedir, aceitar, recusar) está implementada nos agentes e testada, mas as
  condições A e B não a usam: o protocolo de pedidos é o P2 da `feat/protocol-conditions`.
- Com modelos de 7–8B a taxa de ações bem formadas pode ser baixa; é reportada, não escondida.
