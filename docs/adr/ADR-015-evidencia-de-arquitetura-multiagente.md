# ADR-015: Evidência de arquitetura multiagente no código

- **Status:** Accepted
- **Data:** 2026-10-18
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2 (efeito da coordenação), SQ3 (padrões de falha na interação)

## Contexto

A dissertação afirma avaliar protocolos de coordenação em um sistema multiagente baseado em LLM
(ADR-013, ADR-014). Até aqui a arquitetura de cada protocolo estava implícita nos nós do grafo e nas
ferramentas de cada agente, os critérios de agência (C1–C7, `docs/agency_conformance.md`) eram números
sem veredito, e a decisão de um agente antes e depois de um pedido não era registrada. Faltava poder
dizer, pelos dados de cada execução, se aquilo que rodou foi mesmo a arquitetura descrita.

## Decisão

1. **Arquitetura como dado.** `pipeline/protocol_spec.py::PROTOCOL_SPECS` descreve, por protocolo,
   os papéis e os canais (quem pode enviar que ato, com que conteúdo, a quem) e o máximo de pedidos
   por requisito. A estratégia grava a descrição em `manifest["architecture"]`.
2. **Conformidade.** `evaluation/protocol_conformance.py` confere as mensagens de uma execução contra
   o spec: canal e ato permitidos, participantes conhecidos, resposta ligada a um pedido do
   destinatário e limite de pedidos. O resultado vai em `metrics["conformance"]`.
3. **Decisão antes e depois.** O priorizador grava as decisões do primeiro passe; o runner escreve
   `decisions.jsonl` (inicial, final e rótulos de referência). `analyze_protocols` conta, entre as
   decisões alteradas, as que passaram a concordar com a referência, as que deixaram de concordar e as
   neutras, com o exato de McNemar entre melhoras e pioras. É associação dentro da execução, não
   estimativa causal, e não usa a ordem dos rótulos MoSCoW.
4. **Custo por fase.** O custo é separado em trabalho, resposta e reconsideração; a parcela de
   coordenação (tokens fora do primeiro passe) é reportada.
5. **Falhas de interação.** `evaluation/interaction_failures.py` detecta pedido sem resposta,
   resposta que falhou, resposta do classificador ignorada, concordância sem mudança e recusa com
   mudança. Correspondem à família de desalinhamento entre agentes do MAST (Cemri et al., 2025).
   O `DetectorChain` antigo (ADR-003), que nenhum fluxo chama, não foi religado.
6. **Portão de afirmação.** `experiments/architecture_claim.py` decide por braço, com limites
   congelados em `confirmatory.yml::architecture_claim`, se os dados sustentam o rótulo de
   arquitetura multiagente: arquitetura registrada, conformidade sem violações, cada agente com
   variação de ações (entropia > 0) e orçamento esgotado abaixo do limite e, em P2 e P3, o canal
   usado ao menos o mínimo. Vereditos: `supported` (P2, P3), `minimal` (P0, dependência de mão única
   por construção) e `not_supported`. Taxas de aceite e recusa são reportadas e **não** barram o
   veredito (ADR-014: recusa zero é resultado). Se P2 e P3 não usarem o canal, o resultado diz que
   os protocolos são indistinguíveis na prática.
7. **Replay.** `experiments/replay_metrics.py` recalcula atividade, interação, conformidade e falhas
   só de `actions.jsonl`, `messages.jsonl` e `decisions.jsonl` e falha se algo divergir de
   `results.json`.

## Consequências

- A afirmação "avalia arquiteturas multiagente" passa a ser verificável e pode ser negada pelos
  próprios dados.
- Os limites são escolhas do projeto (não vêm da literatura) e ficam no hash do spec: mudá-los invalida
  as execuções confirmatórias anteriores.
- Execuções feitas antes desta mudança não têm `architecture`, `conformance` nem `decisions.jsonl` e
  não passam no portão; todos os braços devem rodar no mesmo commit.
- Fora de escopo, para planos próprios: linha de base de agente único com ReAct, mediador como agente,
  negociação em várias rodadas e controle descentralizado.

## Referências

- Arquivos: `pipeline/protocol_spec.py`, `evaluation/protocol_conformance.py`,
  `evaluation/decisions.py`, `evaluation/interaction_failures.py`, `evaluation/agent_activity.py`,
  `experiments/architecture_claim.py`, `experiments/replay_metrics.py`,
  `experiments/analyze_protocols.py`, `experiments/configs/confirmatory.yml`
- Testes: `test_protocol_conformance.py`, `test_interaction_failures.py`,
  `test_architecture_claim.py`, `test_protocol_analysis.py`, `test_agentic_strategy.py`
- ADRs: ADR-003, ADR-012, ADR-013, ADR-014
- Literatura (conferir nas fontes): Wooldridge & Jennings (1995); Cemri et al. (2025, MAST); Tran et
  al. (2025); FIPA-ACL
