# ADR-014: Protocolo confirmatório dos protocolos de coordenação

- **Status:** Accepted
- **Data:** 2026-10-18
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2 (efeito da coordenação), SQ3 (padrões de falha na interação)

## Contexto

O ADR-013 define os protocolos P0, P2 e P3. Faltava fixar, antes de qualquer execução confirmatória,
como o efeito é medido e testado, e fazer o código recusar o que fugir disso. A fundamentação teórica
(coordenação como gestão de dependências, Malone & Crowston, 1994; autonomia local de agentes ReAct,
Wooldridge & Jennings, 1995; Yao et al., 2023) separa **autonomia local**, igual nos três braços, de
**coordenação global**, o único fator.

## Decisão

- **Fator único:** protocolo de coordenação (P0, P2, P3). O `pipeline` é controle arquitetural
  externo, não um quarto nível: P0 − pipeline troca também a natureza dos componentes e **não** estima
  o efeito da coordenação (`analyze_agency` imprime esse aviso).
- **Desenho congelado** em `experiments/configs/confirmatory.yml` (PrioReSpa, censo N=401, ES/ES,
  mesmo modelo nos braços, T=0, seed 42, uma execução por braço, regime `zero_shot`) **mais uma segunda
  execução idêntica do P0**, o piso de ruído (abaixo). O sha256 do
  arquivo e a lista de desvios vão em `manifest["confirmatory"]`; o analisador recusa execuções
  fora do desenho (`--exploratory` as aceita e marca o resultado). Smoke tests não entram na inferência.
- **Rótulos de referência.** F/NF e MoSCoW do PrioReSpa são critérios automáticos do benchmark, não
  verdade humana independente: concordância é com o benchmark.
- **Métricas.** Macro-F1 e MCC (tipo); Macro-F1 e κ **nominal** (MoSCoW) são primárias, acurácia é
  complementar. O κ ponderado só é primário quando `ORDINAL_MAPPING_VERIFIED` for verdadeiro, o que
  exige demonstrar que os códigos 1..4 do CSV significam Won't..Must; até lá sai marcado como
  exploratório. (Verificação pendente: codebook do Zenodo 10.5281/zenodo.15552978.)
- **Inferência.** Métricas de corpus: teste de permutação pareado por `requirement_id` (troca, por
  requisito, qual protocolo produziu a saída) e IC por bootstrap pareado. Custo por requisito:
  Friedman entre os três protocolos e Wilcoxon por contraste. Completude: McNemar. Contrastes
  planejados P2−P0, P3−P0, P3−P2, com correção de Holm sobre eles além do p bruto.
- **Piso de ruído.** No smoke de 10 itens, P0 repetido deu saídas idênticas, mas P2 e P3, sem nenhum
  pedido trocado, divergiram do P0 em justificativas e em alguns itens: T=0 não torna execuções
  separadas idênticas. Sem uma medida disso, uma diferença P2−P0 não se distingue de ruído. Por isso o
  desenho inclui `P0R`, o P0 rodado de novo sob a configuração idêntica. Ele **não** é nível do
  fator: entra só como contraste P0R−P0 por métrica, e cada contraste de protocolo sai anotado com
  `noise_floor_delta` e `exceeds_noise_floor` (|Δ| do protocolo maior que |Δ| do ruído). É uma única
  estimativa de ruído, não sua distribuição. O analisador confirmatório exige a repetição.
- **Troca de modelo.** O modelo é o campo `model` do spec e nenhum outro arquivo o repete: o
  registro de provedores (`llm/factory.py`, `@provider`) resolve `<provedor>/<nome>`; `run --arm`
  lê modelo, idiomas, seed e dataset do spec e recusa flags que desviem do desenho; `check-model` e a
  checagem automática no início do `run` (preflight: uma chamada simples e uma com ferramenta) falham em
  segundos se o modelo estiver inacessível, sem cota ou sem tool calling. O manifest grava `models`
  (classificador e priorizador). Mudar o modelo muda o hash do spec, o que invalida execuções anteriores.
- **Orçamento de passos.** Cada episódio de agente tem um teto de chamadas ao modelo, e estourá-lo
  faz o item falhar. Com 5 chamadas, 5 de 190 itens do classificador (2,6%) estouraram na primeira
  rodada do P0, e 5 a 7% dos episódios dos smokes já usavam as 5 (o caminho de quem usa todas as
  ferramentas gasta exatamente 5). Subir o teto só afeta quem o esgota; quem termina antes não muda. O
  valor passou a **8** e entrou no spec (`max_steps`), igual em todos os braços: execução agêntica com
  outro valor sai não confirmatória. A decisão veio do piloto e foi tomada **antes** de qualquer coleta
  confirmatória; não foi otimizada sobre o desfecho. A rodada que usou 5 foi descartada.
- **Identificação dos braços.** Cada execução tem pasta própria
  `<estratégia>_<modelos>_<idioma-dados>-<idioma-prompt>_n<N>[_<tag>]_<AAAAMMDD-HHMMSS>` (início, hora
  local; o trace tem o mesmo nome e `manifest`/`results.json` gravam `started_at`/`executed_at` e `finished_at`). A repetição do P0
  é rodada com `--tag p0-repeat` (rótulo congelado no spec, gravado no manifest e no nome da pasta;
  `assess` só o aceita na estratégia do P0). O analisador procura sozinho, em `--results-dir`, a única
  execução confirmatória de cada braço e recusa a análise se faltar uma ou houver duas candidatas; com
  caminhos explícitos, recusa um P0 com rótulo e uma repetição sem o rótulo, o que impede trocar os
  dois.
- **Três dimensões separadas** (concordância, confiabilidade, custo), sem índice escalar. Uma tabela
  de qualidade × custo indica dominância só quando nenhum custo vai no sentido contrário; caso
  contrário, trade-off.
- **Verificação de manipulação.** Cada episódio registra as ações oferecidas (`available_actions`),
  para distinguir ação indisponível (problema de implementação) de ação disponível e não usada
  (resultado). Aceite ou recusa zero é resultado; o prompt não é alterado depois de ver o resultado.
  Cada mensagem trocada (relatório do classificador, pedidos com evidência, respostas) vai para
  `messages.jsonl`, ao lado de `actions.jsonl`, e `analyze_protocols` conta os fluxos por remetente,
  destinatário, ato e conteúdo.

## Consequências

- Uma execução por braço mais a repetição do P0: os IC descrevem variação entre requisitos, não
  entre execuções, e o ruído entre execuções tem uma só estimativa. Custo: +1 execução n=401
  (~75 min). Declarado em `limitations` na saída.
- Um único modelo (`ollama/qwen2.5:7b`, 7B quantizado, T=0) e um único benchmark: sem generalização a
  outros LLMs ou conjuntos. O relatório deve registrar a versão do modelo no Ollama e a data das
  execuções.
- Escolha do modelo: decisão de engenharia experimental, não proposição teórica. O `qwen2.5:7b` é o
  modelo cuja autonomia de priorizador (uso de ferramentas e ações válidas) foi verificada nos smokes
  e roda local, sem custo por chamada. O modelo foi alterado duas vezes **antes de qualquer coleta
  confirmatória** (qwen2.5:7b → GPT-4.1-mini → qwen2.5:7b); a segunda troca ocorreu porque o limite de
  uso da API acabou. Os smokes com GPT-4.1-mini e a rodada do P0 interrompida são exploratórios e ficam
  fora da inferência (o hash do spec mudou, então o analisador os ignora). Não se afirma que o modelo
  seja superior para Engenharia de Requisitos.
- MoSCoW é categorização de prioridade, subconjunto de priorização de requisitos.
- A correção de Holm é decisão do projeto, não está no texto da fundamentação.

## Referências

- Arquivos: `experiments/configs/confirmatory.yml`, `experiments/confirmatory.py`,
  `experiments/analyze_protocols.py`, `evaluation/metrics/corpus.py`,
  `evaluation/metrics/statistical.py`, `evaluation/agent_activity.py`, `domain/agent_actions.py`
- Testes: `tests/unit/test_protocol_analysis.py`
- ADRs: ADR-012, ADR-013
- Literatura (conferir nas fontes): Malone & Crowston (1994); Wooldridge & Jennings (1995); Yao et al.
  (2023); Hong et al. (2024, MetaGPT); Fourney et al. (2024, Magentic-One); Dietterich (1998);
  Demšar (2006); Ojala & Garriga (2010); Dror et al. (2018); Sokolova & Lapalme (2009); Chicco &
  Jurman (2020); Cohen (1968); Wohlin et al. (2024)
