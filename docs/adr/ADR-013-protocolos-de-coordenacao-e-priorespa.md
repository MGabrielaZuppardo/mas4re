# ADR-013: Protocolos de coordenação P0/P2/P3 entre agentes autônomos e o dataset PrioReSpa

- **Status:** Accepted
- **Data:** 2026-10-17
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2 (efeito da coordenação), SQ3 (padrões de falha na interação)

## Contexto

A dissertação estuda o efeito da **estratégia de coordenação** entre classificador e priorizador.
O ADR-012 deu aos dois a forma de agente (laço ReAct, mensagens tipadas, direito de recusar), mas
deixou a capacidade social **sem uso**: `AgenticClassifier.answer_requests` só era chamado em testes,
e a ferramenta `request_reclassification` prometia ao priorizador uma resposta que nenhum grafo
entregava. A Condição B do ADR-009 também não serve ao novo desenho: ela reenvia o lote inteiro ao
classificador, sem dizer o que está errado e sem que ninguém possa recusar, ou seja, é coordenação
entre componentes, não entre agentes.

Falta ainda o dataset da dissertação. O PrioReSpa (401 requisitos em espanhol, com tipo, categoria
PROMISE e **MoSCoW humano**) tem gabarito de prioridade; o PROMISE não tem. Seu gabarito é
desbalanceado (Must 182, Should 205, Could 13, Won't 1; 51% para o baseline de maioria) e
`relevance` quase não se correlaciona com MoSCoW (Spearman −0,06).

## Decisão

**O único fator experimental é o protocolo de coordenação.** Os agentes (modelos, prompts,
temperatura, orçamento de passos, ferramentas de observação) são idênticos nas três condições; muda
o que cada agente **pode fazer** além do trabalho em cada item e quando o ambiente lhe dá a vez
(`pipeline/protocols.py`). A topologia é estática, sem arestas condicionais: um turno em que ninguém
tem nada pendente não faz nada, e cada protocolo troca no máximo **uma rodada** de mensagens.

| Protocolo | Estratégia | Grafo | Quem pode falar com quem |
|---|---|---|---|
| P0 fixo | `pipeline_agentic` | `classifier → prioritizer → cross_check` | Só o `inform` do classificador. O priorizador **não tem** a ferramenta de pedido: a ação não existe |
| P2 pares | `pipeline_agentic_p2` | `… → classifier_answers → prioritizer_reconsiders → cross_check` | O priorizador pode pedir (`request_reclassification`); o classificador decide aceitar (revendo) ou recusar; o priorizador decide **de novo**, uma vez |
| P3 mediado | `pipeline_agentic_p3` | `… → mediator → prioritizer_answers → cross_check` | O mediador pede ao priorizador que reconsidere, com a evidência; o priorizador decide aceitar (com a prioridade revista) ou recusar |

- **P2, reconsideração.** O segundo episódio do priorizador usa um runtime **sem** a ferramenta de
  pedido, de modo que o diálogo é limitado por construção a uma rodada. A decisão anterior e a
  resposta do par entram como crenças do episódio (`phase = reconsideration`).
- **P3, mediador.** É código do ambiente (como o `cross_check`), não um agente: seu único poder é
  enviar um `REQUEST` tipado com evidência. Dois tipos de achado. (a) *Conflito*: categoria NFR
  crítica (SE, PE, A, FT) com Could/Won't, o detector que já existia. (b) *Inflação de Must*: mais de
  60% do backlog como Must (referência do DSDM); os candidatos são os Musts de **menor score**, só o
  excesso, desempatando pela pior posição. Conflito vem antes de inflação, no máximo
  `max_requests = 25` pedidos, e o teto é declarado em `metrics["mediator"]["capped"]`. Quem decide
  é o priorizador; o limite de 60% vem da literatura, não do gabarito do dataset (45% de Must).
- **P1 sai.** Com dois agentes e fluxo de mão única, um protocolo de difusão seria idêntico ao P0.
- **Métricas de interação** (`metrics["interaction"]`: pedidos, aceites, recusas, falhas, taxas) e do
  mediador (`metrics["mediator"]`: achados, pedidos, parcela de Must antes e depois) saem de cada
  execução; as taxas de aceite e recusa são a evidência real do critério C4.

**Dataset e idiomas.** `PrioReSpaAdapter` com `moscow_categorization` 4=Must, 3=Should, 2=Could,
1=Won't; ids de texto (`1035-2` existe). O registro `datasets/registry.py` e `RunConfig.dataset_name`
tiram o runner do acoplamento ao PROMISE. As métricas contra o gabarito ordinal
(`compute_priority_gold_metrics`) são: acurácia ao lado do baseline de maioria, κ ponderado linear,
Spearman, Kendall τ-b, MAE ordinal, precisão/revocação por classe, matriz de confusão e cobertura.

Os prompts existem em PT, EN e **ES** (`Lang.ES`), em dicionários `Lang → texto` lidos por
`localized()`, que **levanta erro** se faltar o idioma, em vez de cair em EN em silêncio. O
`RunConfig` leva dois campos: `lang` (idioma do prompt) e `data_lang` (coluna de texto). O plano de
análise, declarado antes de rodar:

| Célula (dado/prompt) | Papel |
|---|---|
| ES/ES | **primária**, a única confirmatória |
| ES/PT | sensibilidade: isola o idioma do prompt (prompts PT já existem) |
| PT/PT | sensibilidade: exige a base traduzida e validada (`scripts/translate_priorespa.py`, `scripts/validate_translation.py`) |

Cada protocolo é comparado **dentro** de uma célula; idioma nunca entra como fator cruzado. O
`experiments/analyze_agency.py` recusa misturar células e só pareia um workflow com um agêntico que
difere apenas na estratégia.

## Consequências

**Positivas:**
- A pergunta de pesquisa vira um experimento de fator único com quatro braços: `pipeline` (workflow,
  controle de C7), P0, P2 e P3, sobre os mesmos itens, pareados por id.
- Os critérios C4 (recusa), C5 (observação muda a ação) e C7 (ablação) passam a ter números de
  execução real, não só de teste unitário (`experiments/analyze_agency.py`).
- Cada mensagem fica auditável: pedido, resposta (`agree`/`refuse`/`failure`) e a decisão final.

**Negativas / trade-offs aceitos:**
- O diálogo tem **uma** rodada. Isso mede a coordenação por consulta, não negociação; é uma escolha
  de custo e comparabilidade.
- A recusa depende do modelo: com 7–8B pode haver aceite quase sempre, e então a taxa de recusa zero
  é resultado, não bug (C4 exigiria ambas as taxas acima de zero).
- O limite do DSDM refere-se a **esforço**; o mediador o aplica à **contagem de itens**, por não
  haver estimativa de esforço no PrioReSpa. É uma aproximação declarada.
- O mediador usa duas heurísticas fixas (categorias críticas e 60% de Must). Se o gabarito tiver
  menos Must que o limite, a inflação quase nunca dispara no PrioReSpa (45%) e o P3 será dominado
  pelos conflitos; o número de pedidos é reportado para isso ficar visível.
- O gabarito do PrioReSpa tem só 14 Could/Won't: as métricas por classe dessas duas são instáveis e a
  conclusão se apoia em acurácia, κ ponderado e correlações.
- A base PT é tradução automática; a diferença ES/ES × PT/PT mistura dado, tradução e prompt, e a
  dissertação não deve atribuí-la só à tradução. O desenho 2×2 é incompleto (falta dado PT + prompt ES).
- Mais chamadas de LLM por item que o workflow; custo e cobertura vão sempre ao lado da qualidade.

**Neutras:**
- A Condição B do ADR-009 e o workflow do ADR-011 permanecem como controles, sem mudança.
- Os arquivos do SBQS (grade de 18 condições, scripts de bootstrap, dados arquivados) foram removidos
  desta branch; continuam em `develop` e em `artifact/sbqs-camera-ready`.

## Alternativas consideradas

- **Reaproveitar o reenvio cego da Condição B** — descartada: sem evidência e sem direito de
  recusa não é coordenação entre agentes.
- **Mediador que é um LLM** — descartada: acrescentaria um terceiro agente e uma segunda fonte de
  variação; o mediador do ambiente isola o efeito da coordenação.
- **Negociação em várias rodadas / Contract Net** — descartada por custo e por tornar os protocolos
  incomparáveis em número de chamadas; fica como trabalho futuro.
- **Idioma como segundo fator** — descartada: a dissertação tem um eixo só (coordenação); o idioma
  entra como replicação.

## Impacto na pesquisa

Responde à pergunta central (como estratégias de coordenação centralizada influenciam classificação
e priorização) com agentes que se qualificam como tais. Hipóteses a testar por célula, com
Δ pareado e IC por bootstrap: P2 e P3 versus P0 em acurácia de tipo, acurácia MoSCoW e κ ponderado,
e a taxa de aceite/recusa como mecanismo. O regime de informação continua `zero_shot`: a memória
guarda só respostas do próprio agente e o mediador não lê o gabarito.

## Referências

- Arquivos afetados: `pipeline/protocols.py`, `pipeline/mediation.py`,
  `pipeline/nodes/protocol_nodes.py`, `agents/agentic/prioritizer.py`, `agents/agentic/classifier.py`,
  `domain/messages.py`, `domain/agent_actions.py`, `domain/enums.py`, `prompts/localization.py`,
  `prompts/v2/`, `datasets/priorespa.py`, `datasets/registry.py`, `datasets/translator.py`,
  `evaluation/metrics/prioritization.py`, `evaluation/metrics/statistical.py`,
  `evaluation/agent_activity.py`, `experiments/strategy.py`, `experiments/runner.py`,
  `experiments/analyze_agency.py`, `cli/main.py`
- Testes: `tests/unit/test_protocols.py`, `test_mediation.py`, `test_analyze_agency.py`,
  `test_priorespa_adapter.py`, `test_priority_gold_metrics.py`, `test_prompt_languages.py`
- ADRs relacionados: ADR-009 (Condição B), ADR-011, ADR-012 (deixou a capacidade social sem uso;
  este ADR a liga)
- Literatura a citar (**conferir nas fontes antes da dissertação**; vieram de memória): FIPA Request
  Interaction Protocol (SC00026); Smith (1980), Contract Net Protocol; Malone & Crowston (1994),
  *The interdisciplinary study of coordination*, ACM Computing Surveys; DSDM (limite de ~60% de
  esforço em Must); Wooldridge & Jennings (1995)
