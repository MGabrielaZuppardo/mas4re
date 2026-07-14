# Contexto — Artigo SBQS (Trilha Técnica)

> Documento de contexto consolidado a partir de sessão de planejamento em 2026-07-13.
> Objetivo: reunir tudo que já foi decidido/verificado para retomar o trabalho do artigo
> sem precisar re-derivar o raciocínio.

---

## 1. Ideia do artigo

**Título de trabalho:**
"Pipeline Multi-Agente vs. Agente Único para Classificação de Requisitos: Um Estudo de
Ablação Controlado com LLMs Abertos"

**Por que este escopo (e não um mais ambicioso):**
- Restrito à etapa de **classificação RF/RNF** — é a única etapa com dados robustos e
  ground truth confiável (dataset PROMISE NFR+, ~625 requisitos, PT/EN). Priorização e
  NFRIC ficam fora: priorização não tem avaliação de qualidade coletada nesse grid, e
  `datasets/nfric.py` ainda é stub.
- A afirmação central não é "o MAS é sempre melhor" (frágil — pode não ser significativo
  em todas as condições). É: **"o pipeline supera o agente único, e a ablação isola se o
  ganho vem do contrato de estado tipado entre agentes ou apenas da especialização de
  prompt"**. Essa formulação é defensável nos dois desfechos possíveis:
  - Se significativo em ≥4/6 condições → contrato tipado agrega valor além do prompt.
  - Se não significativo → ganho é atribuível a especialização de prompt; narrativa
    ainda é um achado válido, não um fracasso do artigo.

## 2. Perguntas de pesquisa

- **RQ1**: O pipeline MAS (classificador especializado) supera um agente único de mesmo
  LLM na tarefa de classificação RF/RNF?
- **RQ2**: O ganho observado vem do contrato de estado tipado entre agentes, ou é
  replicável apenas com prompts especializados em duas chamadas ao mesmo LLM (ablação
  two-call)?
- **RQ3** (secundária, dados já disponíveis sem custo extra): O ganho é uniforme entre
  modelos, ou depende da capacidade do LLM base? (ver achado da Seção 4 — não é uniforme).

## 3. Desenho experimental

- **3 LLMs abertos locais via Ollama**: qwen2.5:7b, llama3.1:8b, mistral:7b
- **2 idiomas**: PT / EN
- **3 estratégias**: `baseline` (agente único), `pipeline` (MAS, classificador
  especializado), `two_call_baseline` (ablação: 2 chamadas ao mesmo LLM, sem contrato de
  estado tipado)
- Total: 18 condições (3 modelos × 2 idiomas × 3 estratégias)
- **Dataset**: PROMISE NFR+ completo (n=625, dropa nulos → ~623-625 dependendo da run),
  seed=42, temperatura=0.0
- **Testes estatísticos**: Wilcoxon signed-rank pareado por item (pareado por **texto**,
  não por `id` — ver Seção 5) + Cohen's h como effect size, correção de Bonferroni
  α' = 0.05/6 = 0.0083

## 4. Status dos dados (em 2026-07-13, grid rodado hoje)

- Grid iniciado em `experiments/results/grid_run_18cond_20260713T080325.log`, PID em
  `experiments/results/grid_run_18cond.pid`.
- **14/18 condições concluídas** no momento desta sessão (faltam as 3 condições de
  `mistral:7b` EN + repetição).
- Resultados agregados em `experiments/results/grid_summary.csv`.
- Accuracies observadas (classificação, dados parciais):

  | Modelo | Idioma | Baseline | Pipeline | Two-call |
  |---|---|---|---|---|
  | qwen2.5:7b | PT | 0.742 | 0.814 | 0.812 |
  | qwen2.5:7b | EN | 0.682 | 0.789 | 0.811 |
  | llama3.1:8b | PT | 0.645 | 0.693 | 0.685 |
  | llama3.1:8b | EN | 0.684 | 0.726 | 0.736 |
  | mistral:7b | PT | 0.598 | 0.569 (⚠ pipeline < baseline) | — |
  | mistral:7b | EN | — (pendente) | — (pendente) | — (pendente) |

- **Achado RQ3 já visível**: o ganho do pipeline não é uniforme — grande em qwen
  (+7-10pp), moderado em llama (+4-5pp), **negativo em mistral/PT** (pipeline pior que
  baseline). Isso deve ser reportado como achado honesto, não escondido — reforça a tese
  de que o ganho depende da capacidade do modelo base.

## 5. Investigação de qualidade de dados (bug histórico + verificação atual)

### 5.1 Bug conhecido do artigo anterior (ISE'26, já submetido — texto travado)
Ver memória `project_parse_failure_bias_post_submission`: `agents/classifier.py::_parse_response`
antigamente fazia fallback silencioso para `RequirementType.FUNCTIONAL` com
`confidence=0.0` em caso de JSON não parseável, e os scripts de estatística da época
contavam esse fallback como predição real (geralmente errada) em vez de excluir. Corrigido
em 2026-07-12 no pipeline de classificação (`domain/models.py` / `agents/classifier.py` /
`evaluation/metrics/classification.py`), mas **não retroativamente** nos scripts que
geraram os números do artigo já submetido.

### 5.2 Verificação nos dados do grid atual (2026-07-13) — bug NÃO recorreu
- Todos os `results.json` das 14 condições concluídas já carregam o campo `parse_failed`.
- Auditoria confirmou `parse_failed=0` e `confidence==0.0` count = 0 em todas as 14
  condições — mas isso é **enganoso à primeira vista** (ver 5.3).

### 5.3 Mecanismo real do "shortfall" de itens (n < 625 em alguns runs)
- `Requirement.id` é um `uuid4()[:8]` gerado a cada `load()` (`domain/models.py:16`) —
  **nunca estável entre execuções diferentes**. Pareamento correto deve ser sempre por
  **texto**, nunca por id.
- Em `agents/classifier.py:144-145`: itens cuja severidade máxima de falha é `FATAL`
  (JSON não parseável, `parse_failed=True`) são **excluídos inteiramente** da lista de
  predictions (`continue`, nunca entram em `results`) — por isso nunca aparecem com
  `parse_failed=True` dentro de `predictions`: eles simplesmente não existem lá.
- Isso explica o shortfall observado:
  - `baseline_llama3.1-8b_en`: 17 FATAL → 625−17=608 predictions
  - `pipeline_mistral-7b_pt`: 17 FATAL → 625−17=608 predictions
  - `baseline_qwen2.5-7b_pt`: 1 FATAL → 625−1=624 predictions
  - Diferença de 2 em alguns runs qwen (623) vem de **2 linhas com texto duplicado no
    próprio CSV do PROMISE** — característica do dataset, não bug.
- `state.errors` fica vazio (`[]`) porque esse campo só registra falhas de *retry
  esgotado* (erro de rede/timeout), não falhas de parse — via de exclusão diferente e
  não relacionada.

### 5.4 Conclusão: não precisa re-rodar as 18 condições
- A exclusão de itens `FATAL` já acontece **antes** da escrita do `results.json` — os
  dados coletados já estão corretos, não contaminados.
- Os scripts de estatística já pareiam corretamente por **texto** e por **interseção**
  de itens parse-válidos entre as duas pernas de cada comparação:
  - `experiments/compute_stats.py::load_all_runs` indexa por `text`; `run_rq1` faz
    `common = sorted(set(base_run) & set(pipe_run))` antes do Wilcoxon.
  - `experiments/compute_ablation_stats.py::_binary_outcomes` + interseção de
    `abl_outcomes`/`pip_outcomes` faz o mesmo para two_call vs pipeline.
- **O que falta não é correção, é transparência**: a taxa de exclusão por parse-failure
  não aparece hoje em nenhum output agregado (só é visível cavando `failure_records`
  dentro de cada `results.json`). Recomendação: escrever um pequeno script de auditoria
  que produza uma tabela `condição | n_dataset | n_fatal_excluded | %` a partir de
  `failure_records` com `severity=fatal` — vira tabela suplementar / nota de rodapé no
  artigo, documentando a taxa de parse-failure por modelo/estratégia como ameaça à
  validade conhecida e quantificada (ainda **não escrito** nesta sessão).

## 6. Contribuição metodológica extra (diferencial do artigo)

Transformar a investigação da Seção 5 em uma pequena seção metodológica do próprio
artigo ("lições de avaliação" / nota sobre qualidade de dados): taxa de exclusão por
parse-failure documentada por condição, pareamento por interseção de itens válidos por
texto (não id). É o tipo de rigor que revisores de qualidade de software valorizam, e
transforma um problema encontrado durante o desenvolvimento em contribuição —
sem tocar no artigo ISE'26 já submetido.

## 7. Ameaças à validade (mapeadas)

- Apenas 1 dataset (PROMISE NFR+), 1 repetição por condição (sem múltiplas seeds ainda).
- Apenas modelos abertos locais via Ollama — não testado contra modelos proprietários
  maiores (Claude, GPT).
- Taxa de parse-failure variável por condição (0%–2.7% observado) — documentada, não
  escondida.
- `n` de itens pareados varia por comparação (efeito da exclusão FATAL diferente por
  condição) — reportar `n_pairs` explicitamente em cada linha de resultado, não só o
  `n` nominal do dataset.

## 8. Enquadramento no CFP do SBQS

- **Tag primária**: Engenharia de Requisitos (tema central: classificação RF/RNF).
- **Tag secundária**: Inteligência Artificial Generativa e Qualidade de Software (método:
  LLMs generativos em arquitetura multi-agente, avaliação empírica de qualidade de
  output).
- **Tag opcional (mais forçada)**: Verificação, Validação e Testes — só se o CFP exigir
  múltiplas tags; o enquadramento é pelo ângulo metodológico (ablação controlada,
  pareamento estatístico rigoroso, auditoria de parse-failure), não pelo tema central.

## 9. Próximos passos (ainda não executados nesta sessão)

1. Aguardar conclusão das 4 condições restantes do grid (`mistral:7b` EN).
2. Escrever script de auditoria de taxa de parse-failure por condição (Seção 5.4).
3. Rodar `compute_ablation_stats.py` e `compute_stats.py`/RQ1 sobre as 18 condições
   completas para obter Wilcoxon + Cohen's h finais por condição.
4. Montar esqueleto do artigo (introdução, RQs, métodos, seção de qualidade de dados,
   resultados, ameaças à validade) — ainda não iniciado.
