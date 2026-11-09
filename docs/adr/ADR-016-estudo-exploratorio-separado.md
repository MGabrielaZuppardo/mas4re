# ADR-016: Estudo exploratório separado do desenho confirmatório

- **Status:** Accepted
- **Data:** 2026-10-19
- **Autor:** Gabriela Zuppardo
- **SQ relacionada:** SQ2 (efeito da coordenação), SQ3 (padrões de falha na interação)

## Contexto

A análise confirmatória (ADR-014) deu um resultado nulo: os contrastes P2−P0 e P3−P0 ficaram do tamanho
do piso de ruído; o mediador do P3 não enviou nenhum pedido (nada passou do limite de Must nem havia
conflito de categoria crítica) e o P2 enviou 6. Ao investigar o piso de ruído apareceu uma dependência de
caminho: a memória episódica do priorizador só é escrita nos 10 primeiros itens (aquecimento) e depois
fica congelada, e a visão do backlog (`backlog_overview`) congela as prioridades decididas até ali. As
quatro execuções do qwen2.5:7b caíram em dois regimes de Must (110 e ~194 de 398), e as duas com o mesmo
aquecimento (P0 repetido e P3) deram saídas quase idênticas. A diferença entre protocolos fica
confundida com o regime que cada execução sorteou.

## Decisão

Um estudo exploratório à parte, que não altera o `confirmatory.yml` (o hash do spec é guardado por um
teste), nem as execuções confirmatórias, nem os analisadores confirmatórios.

- **Separação.** `python -m cli.main explore --variant ID --arm ARM`. As saídas vão para
  `experiments/exploratory/{results,traces,analysis}/`, então `find_arms` e `analyze_protocols` nunca as
  veem. O manifest traz o bloco `exploratory` (variante, hipótese, hash do `exploratory.yml`, execução
  de onde o aquecimento foi copiado) e `confirmatory.is_confirmatory=false` com o desvio "exploratory
  variant". As variantes e as hipóteses ficam em `experiments/configs/exploratory.yml`, escritas antes
  de rodar.
- **Variantes.** X0 (sem execuções novas): aquecimento × regime de Must e contrastes tendo o P0 repetido
  como baseline. X1: P0, P2 e P3 com o aquecimento fixado. X2: P3 com o mediador ativo (limite de Must em
  0,40, escolhido depois de ver as execuções confirmatórias). X3: `foundry/gpt-4.1-mini` em P0, P0 repetido, P2 e P3, em célula própria. X4: `ollama/llama3.1:8b` em P0 e P0 repetido (P2 e P3 só se
  justificarem), célula própria, sem custo. Expectativa declarada: pouca variação de ações no priorizador, o que
  derruba o portão de arquitetura e é um achado sobre a dependência da agência em relação ao modelo.
- **Hash por variante.** O `spec_sha256` do manifest exploratório é o hash do bloco da variante, para que
  acrescentar outra variante não mude o das demais. A execução X3/P0 gravou o hash do arquivo inteiro, antes
  dessa mudança; o bloco do X3 não foi alterado.
- **Aquecimento fixado.** `WarmupSeed` pré-carrega a memória do priorizador e o contador de prioridades
  decididas com as 10 primeiras decisões de uma execução de referência (texto, prioridade da primeira
  passada e justificativa) e congela. Os 10 primeiros itens ficam, na prática, copiados da referência.
  Só o priorizador é fixado: a classificação variou em no máximo 6 itens entre execuções.
- **Mediador parametrizável.** `must_limit` (padrão 0,60) chega ao `mediate`; o valor usado vai em
  `metrics["mediator"]`.
- **Análise.** `python -m experiments.analyze_exploratory --warmup | --variant ID`. Descritiva: tamanho
  de efeito com IC de bootstrap ao lado do piso de ruído, sem p-valor nem correção de multiplicidade, e
  um aviso fixo no JSON. Recusa comparar modelos diferentes.

## Consequências

- Os achados exploratórios geram hipóteses e não entram como evidência confirmatória; qualquer confirmação
  exigiria um novo desenho congelado antes da coleta.
- Cada variante tem uma execução por braço (R=1) e, no caso do qwen, o mesmo benchmark e rótulos
  automáticos; as limitações do ADR-014 valem aqui também.
- Com o aquecimento fixado, os 10 primeiros itens não são independentes da referência: as comparações
  dizem respeito aos itens seguintes.
- O limite 0,40 do mediador foi escolhido a partir dos dados confirmatórios; isso é decisão guiada
  pelos dados e só serve para um estudo exploratório.

## Referências

- Arquivos: `experiments/configs/exploratory.yml`, `experiments/exploratory.py`,
  `experiments/analyze_exploratory.py`, `agents/memory.py` (`WarmupSeed`),
  `agents/agentic/prioritizer.py`, `pipeline/mediation.py`, `cli/main.py` (`explore`)
- Testes: `tests/unit/test_exploratory.py`, `tests/unit/test_protocols.py::TestMediatorLimit`
- ADRs: ADR-012, ADR-013, ADR-014, ADR-015
