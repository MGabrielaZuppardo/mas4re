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
| C8 | A execução obedeceu à arquitetura do protocolo | Canais e atos permitidos descritos em `pipeline/protocol_spec.py`; `evaluation/protocol_conformance.py` confere as mensagens (canal, participante, resposta ligada a pedido, limite por requisito) | mensagens fora do spec: a arquitetura que rodou não é a descrita |
| C9 | A interação mudou decisões, e isso é mensurável | `decisions.jsonl` (decisão inicial e final) e `analyze_protocols` (`interaction_effect`); falhas em `interaction_failures` | o canal existe mas nada muda: protocolo indistinguível do P0 |
| C7 | Ablação contra o workflow nos mesmos itens | `pipeline` (workflow) × `pipeline_agentic`, com custo, mesmo que o ganho seja nulo | a comparação não existe |

## Estado

| # | CI (LLM roteirizado) | Execução real |
|---|---|---|
| C1 | `test_agency_conformance.py::TestC1…` | `agent_activity.action_sequence_entropy_bits` no `results.json` |
| C2 | `TestC2…` | `agent_activity.budget_exhaustion_rate` |
| C3 | `TestC3…` | n/a (propriedade do código) |
| C4 | `TestC4…`, `test_protocols.py` | `metrics.interaction` (`agree_rate`, `refuse_rate`) de `pipeline_agentic_p2` (o classificador responde ao priorizador) e `pipeline_agentic_p3` (o priorizador responde ao mediador); consolidado por `analyze_agency.py` |
| C5 | `test_agentic_runtime.py::TestTheModelDecides` | `analyze_agency.py`: tabela de contingência observação × ação seguinte (Fisher ou χ², V de Cramér), ou `insufficient_data` |
| C6 | `TestC6…` | inspeção de `actions.jsonl` |
| C7 | `test_analyze_agency.py` (a análise, em dados sintéticos) | `analyze_agency.py --workflow`: `pipeline` × `pipeline_agentic*` pareados por id (McNemar, Δ de acurácia e κ ponderado com IC por bootstrap), com custo e cobertura ao lado |

As execuções reais de C4, C5 e C7 seguem o plano de células de idioma do ADR-013: ES/ES (primária) e
as demais como sensibilidade; o script recusa misturar células.

## O que o código decide e o que o modelo decide

| Decidido pelo código (é o ambiente) | Decidido pelo LLM (é o agente) |
|---|---|
| Quais itens chegam, o conjunto de ferramentas, o orçamento de passos, a validação de esquema, de quem é a vez | Qual ferramenta chamar, se chama alguma, em que ordem, quantas vezes, quando concluir, o que dizer ao par, se aceita ou recusa um pedido |

Proibido no código dos agentes: `if confiança < x: verificar`, chamadas fixas de ferramenta e um
procedimento ("consulte a taxonomia e depois revise") no prompt. A política de retenção da memória
(só respostas confiantes e consistentes viram exemplo) é higiene de memória do ambiente, não controle
de fluxo do agente.

## Limites conhecidos

- A capacidade social está ligada nos protocolos P2 (pedidos entre pares) e P3 (pedidos de um
  mediador do ambiente) do ADR-013; no P0 ela não existe por construção (o priorizador não tem a
  ferramenta de pedido). Cada protocolo troca **uma** rodada de mensagens: mede consulta, não
  negociação.
- C4 só se sustenta se aceite **e** recusa ocorrem em execução real. Com modelos de 7–8B pode haver
  aceite quase sempre; então a taxa de recusa zero é o resultado, e o texto deve dizê-lo.
- C5 e C7 dependem de contagens: abaixo do mínimo (`MIN_OBSERVATIONS`, 20) o script devolve
  `insufficient_data` em vez de um valor-p.
- Com modelos de 7–8B a taxa de ações bem formadas pode ser baixa; é reportada, não escondida.
