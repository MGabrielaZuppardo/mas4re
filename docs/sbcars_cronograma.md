# MAS4RE → SBCARS 2026 — Cronograma & Progresso

> Rastreador vivo. Legenda: ✅ feito · 🟡 em andamento · ⏭️ pendente · ⚠️ risco
> Datas-âncora: resumo JEMS **19/jun** · submissão **26/jun**

---

## Visão geral por semana

| Sem | Janela | Foco | Status |
|---|---|---|---|
| S1 | 11–17/mai | Fundação + Pipeline mínima | ✅ concluída |
| S2 | 18–24/mai | Runner + Strategy + CLI | ✅ concluída |
| S3 | 25–31/mai | Instrumentação SQ3 | 🟡 em andamento |
| S4 | 01–07/jun | Grid + análise | ⏭️ pendente |
| S5 | 08–14/jun | Maratona de texto 1 + Zenodo | ⏭️ pendente |
| S6 | 15–21/jun | Maratona de texto 2 + anonimização | ⏭️ pendente |
| S6.5 | 22–26/jun | Submissão | ⏭️ pendente |

---

## ✅ S1 — Fundação (11–17/mai) — CONCLUÍDA

### Código
- ✅ PR #14 `typed-core`: `Lang` enum, `BaseAgent` genérico, mypy
  strict em domain/agents, rename `priority_justification`
- ✅ PR #15 `langgraph-minimal`: `StateGraph` classifier→prioritizer,
  smoke E2E mockado
- ✅ Infra: pre-commit (ruff 0.15.12), `.gitattributes` LF, CI verde

### Escrita
- ✅ `mas4re.bib` (15 refs validadas)
- ⏭️`main.tex` template SBCARS compilando
- ✅ §1 Introduction (v0, 5 parágrafos) no Overleaf
- ⏭️ §5.1 Research Questions + §5.2 Dataset — **redigidos, faltam
  colar no Overleaf**

---

## ✅ S2 — Runner + CLI (18–24/mai) — CONCLUÍDA

### Código
- ✅ PR #16 `experiment-runner`: `OrchestrationStrategy` + `Baseline/
  PipelineStrategy` + `RunConfig` + `ExperimentRunner` + `manifest.json`
- ✅ PR #17 `runner-metrics`: métricas (classificação/subcategoria/
  MoSCoW) + `results.json`
- ✅ PR #18 `cli-typer`: `mas4re run/eval/compare`; scripts imperativos
  removidos
- ✅ **E2E REAL validado** (Ollama): `mas4re run --strategy pipeline
  --n 10` → **accuracy 0.90 · F1-macro 0.89 · MCC 0.80** (preliminar)

### Escrita
- ⏭️ §5.1/§5.2 no Overleaf (carregado para S3 como background)

---

## 🟡 S3 — Instrumentação SQ3 (25–31/mai) — EM ANDAMENTO

### Código
- ✅ PR #19 `failure-taxonomy`: `FailureMode`, `FailureSeverity`,
  `FailureRecord`, `BatchResult[T]` (ADR-003) — **mergeado**
- ✅ PR #20 `failure-detectors`: `DetectorChain` v1 (schema,
  hallucination, low-confidence, grounding) — **mergeado**
- 🟡 PR #21 `cross-agent-check`: conflito inter-agente +
  `cross_check_node` — **aplicado, em validação/merge**
- ⏭️ PR #22 `trace-writer`: `TraceWriter` JSONL + DI no `BaseAgent`

### Escrita (background)
- ⏭️ Colar §5.1 + §5.2 no Overleaf
- ⏭️ Coletar ref slot (b) — failure modes em LLM (para §3)
- ⏭️ Outline §4 Architecture
- ⏭️ Report ao orientador (S1+S2; destacar E2E real F1=0.89)

---

## ⏭️ S4 — Grid + Análise (01–07/jun)

### Código / execução
- ⏭️ Rodar **grid completo**: 3 modelos × 2 idiomas × 2 arquiteturas
  = **12 condições** (`mas4re run` em lote)
- ⏭️ Notebooks `sq1_per_task.ipynb`, `sq2_comparison.ipynb`,
  `sq3_failure_patterns.ipynb`
- ⏭️ Exportar tabelas LaTeX (`pandas.to_latex`)
- ⏭️ Análise estatística: Wilcoxon pareado + Cliff's δ + bootstrap

### ⚠️ Decision point — 07/jun
Se o grid não fechar → **pivot para Artigo de Dados (4 pág.)**.

---

## ⏭️ S5 — Maratona de Texto 1 (08–14/jun)

- ⏭️ §2 Background
- ⏭️ §3 Related Work (triade NICE/Zadenoori/multi-agente)
- ⏭️ §4 MAS4RE Architecture (+ Fig 1 arquitetura, Fig 2 LangGraph)
- ⏭️ §5.3 Models, §5.4 Metrics, §5.5 Statistical analysis
- ⏭️ §6.1 Results RQ1 (com tabelas reais)
- ⏭️ **Publicar Zenodo → obter DOI**; re-incluir Contribuição 3 em §1
- ⏭️ Report ao orientador

---

## ⏭️ S6 — Maratona de Texto 2 + Anonimização (15–21/jun)

- ⏭️ §6.2 Results RQ2 (custo, Pareto)
- ⏭️ §6.3 Results RQ3 (padrões de falha — diferencial)
- ⏭️ §7 Discussion + Threats to Validity
- ⏭️ §8 Conclusion + Future Work
- ⏭️ Artifact Availability (DOI Zenodo)
- ⏭️ Abstract final + Keywords
- ⏭️ §1 Introduction v2 (à luz dos resultados)
- ⏭️ Grammarly + revisão externa + anonimização (`anonymous,review`)
- ⏭️ **📌 19/jun: registrar resumo no JEMS3**
- ⏭️ Report ao orientador

---

## ⏭️ S6.5 — Submissão (22–26/jun)

- ⏭️ 22/jun: incorporar feedback do orientador
- ⏭️ 23–24/jun: passada final de prosa
- ⏭️ 25/jun: PDF final + verificar Zenodo anonimizado
- ⏭️ **📌 26/jun: SUBMISSÃO JEMS3**

---

## Resumo executivo

### O que JÁ foi feito
- **Código S1+S2 completo**: typed-core → LangGraph → Strategy/Runner
  → métricas → CLI, tudo mergeado, CI verde
- **Pipeline E2E real funcionando** com Ollama (F1-macro 0.89 preliminar)
- **S3 60%**: taxonomia + detectores + cross-agent (PR #19/#20 ✅,
  #21 em merge)
- **Escrita**: §1 no Overleaf; §5.1/§5.2 redigidos; bib com 15 refs
- **Infra robusta**: pre-commit, CI alinhado, manifest reprodutível

### O que FALTA fazer
- **S3**: fechar PR #21 + PR #22 (TraceWriter)
- **S4**: rodar grid 12 condições + notebooks + estatística
- **S5–S6**: redigir §2–§8 + Artifact Availability + Abstract;
  publicar Zenodo; anonimizar
- **Pendências soltas**: colar §5 no Overleaf; ref slot (b);
  Reports ao orientador
- **S6.5**: registrar resumo (19/jun) + submeter (26/jun)

### Indicadores
- Progresso código: **~70%** (S1+S2 ✅, S3 ~60%, S4 0%)
- Progresso escrita: **~12%** (1,5 de ~10 págs)
- Dias até resumo JEMS: contar a partir de hoje até 19/jun
- Dias até submissão: contar a partir de hoje até 26/jun
