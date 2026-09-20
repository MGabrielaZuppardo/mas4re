# Arquivo: resultados anteriores ao ADR-011 (arquivado em 2026-09-20)

## Por que existe

Marca a fronteira entre duas gerações do sistema:

| Geração | Agentes | Onde ficam os resultados |
|---|---|---|
| **Histórica** (este arquivo) | uma chamada ao LLM por item, sem memória, sem ferramenta e sem autocrítica; prompts `v1` | `experiments/results/archive_pre_adr011_20260920/` |
| **Ativa** | agentes do ADR-011 (memória episódica, ferramenta de taxonomia, autocrítica com gatilho, após a correção do crítico) | `experiments/results/` |

Todos os números do artigo submetido vêm desta geração. Nada aqui deve ser apagado nem misturado
com execuções novas: para analisar uma geração por vez, os scripts de análise recebem
`--generation historical|active` (ver `experiments/analysis_common.py::GENERATIONS`).

## Conteúdo

| Grupo | Itens | Período | Commit registrado no manifest |
|---|---|---|---|
| `baseline_*_n625_*` | 8 execuções | 2026-05-24 a 2026-06-20 | `c78f3a8c` (6); `unknown` (2, provavelmente rodadas em container) |
| `pipeline_*_n625_*` | 8 execuções | 2026-05-24 a 2026-06-20 | idem |
| `ablation_20260601T232947/` | 6 execuções `two_call_baseline` + resumo | 2026-06-02 | `c78f3a8c` |
| `grid_summary_nfull_20260524T184454.csv` | resumo do grid principal | 2026-05-24 | — |
| `statistical_results.json`, `rq3_kappa_bootstrap_*.json`, `subcategory_errorprop.json`, `moscow_kappa_comparison.csv` | estatísticas do artigo | 2026-06/07 | — |
| `_run_*.log`, `grid_run_18cond_*.log` | logs de console (**não versionados**, ver avisos) | 2026-07-13 a 2026-07-27 | — |
| `ab_runs_from_logs.json` | resumo estruturado dos `_run_*.log` | gerado em 2026-09-20 | — |

Os modelos são `qwen2.5:7b`, `llama3.1:8b`, `mistral:7b` (Ollama) e `gpt-4.1-mini` (Foundry), em PT e EN.

## A Condição B de julho só existe nos logs

Em 25-27/07/2026 a Condição B (`mediated_pipeline`) rodou com n=625, mas com uma implementação
**anterior** à atual: o reenvio era **por item** (`n_requirements_resent` conta só os itens em
conflito), e não em lote como no ADR-009. As pastas de resultado dessas execuções ficaram vazias e
foram removidas em 2026-09-20; o único registro é o "Run done" impresso em cada
`_run_*.log`, extraído para `ab_runs_from_logs.json`
(`python -m experiments.extract_ab_runs_from_logs`).

Leitura cuidadosa:

- O nome do arquivo não traz o modelo das execuções `*_rep2`. Pelo perfil de MoSCoW (fatia de
  `Must` de 44 a 52%), elas parecem ser `gpt-4.1-mini`, mas isso é **inferência**, não consta no log.
- Nas execuções locais o reenvio nunca "resolveu" um conflito (`n_conflicts_resolved_after_retry`
  igual a 0). O único caso com conflitos resolvidos (2 de 3) é a `_run_b_en_rep2`, coerente com a
  re-amostragem de um modelo de API.
- `_run_b_en_llama_rep2.log` não chegou ao "Run done" (execução incompleta).
- Repetições do mesmo modelo/idioma podem diferir: `qwen2.5:7b`/EN deu 0,7776 e 0,792 sem nenhum
  conflito, enquanto `qwen2.5:7b`/PT deu exatamente 0,808 nas duas.

## Avisos

1. **Os logs não estão no git** e contêm caminhos pessoais da máquina de origem (por exemplo em
   avisos de bibliotecas). Não os versionar em um repositório que precise ser anonimizado; o que é
   útil deles está em `ab_runs_from_logs.json`.
2. Estes resultados têm o **viés de falhas de parse silenciosas** já documentado: quando o JSON do
   modelo não pôde ser lido, o agente devolvia `F` com confiança 0,0. Em `_run_a_en_rep2.log` e
   `_run_b_en_rep2.log` isso aparece 24 e 71 vezes no log.
3. As execuções `unknown` no commit do manifest não permitem reconstruir o código exato.

## Como reproduzir os números do artigo

Da raiz do repositório:

```bash
python experiments/compute_stats.py
python experiments/compute_ablation_stats.py --ablation experiments/results/archive_pre_adr011_20260920/ablation_20260601T232947/ablation_summary.csv
make test-regression
```

Os scripts leem este diretório por `experiments/paths.py::HISTORICAL_RESULTS_DIR`.
