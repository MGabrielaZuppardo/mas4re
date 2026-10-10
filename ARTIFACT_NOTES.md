# Artifact notes

This branch is the artifact for the paper *Beyond Accuracy: Disentangling Prompt
Specialisation from Typed-State Quality Assurance in Multi-Agent LLM Pipelines*. It
holds the code that produced the reported results, the raw predictions, and the
scripts that compute every table.

## 1. What is in this branch

| Path | Content |
|---|---|
| `agents/`, `pipeline/`, `domain/`, `llm/`, `prompts/v1/`, `datasets/` | The three architectures (baseline, two-call baseline, typed pipeline) and their infrastructure |
| `experiments/run_*.py` | Experiment drivers (see section 4) |
| `experiments/compute_*.py` | Analysis scripts, one per table (see section 5) |
| `experiments/results/` | `manifest.json` + `results.json` (every prediction) per run, and the JSON outputs of the analysis scripts |
| `tests/unit/` | Unit tests (`pytest tests/unit`) |

## 2. Which runs produced which numbers

| Result set | Conditions | Executed | Location |
|---|---|---|---|
| Main grid: baseline and pipeline | 3 models x 2 languages x 2 architectures = 12 | 24-25 May 2026 | `experiments/results/{baseline,pipeline}_ollama-*`, summary in `grid_summary_nfull_20260524T184454.csv` |
| Ablation: two-call baseline | 3 x 2 = 6 | 1-2 Jun 2026 | `experiments/results/ablation_20260601T232947/two_call_baseline_*` |
| Repeat of pipeline and two-call (run-to-run variation) | 12 | 6-7 Oct 2026 | `experiments/results/repeat_run2/` |
| Supplement, `gpt-4.1-mini`: baseline and pipeline PT | 2 | 20 Jun 2026 | `experiments/results/{baseline,pipeline}_foundry-*_pt_*` |
| Supplement, `gpt-4.1-mini`: two-call PT and EN | 2 | 15 Sep 2026 | `experiments/results/two_call_baseline_foundry-*` |
| Supplement, `gpt-4.1-mini`: baseline and pipeline EN | 2 | 10 Oct 2026 | `experiments/results/foundry_en_rerun/` |

Notes on provenance:

- The main grid and the ablation were executed with the agent implementations contained
  in this branch. The `gpt-4.1-mini` PT runs (20 Jun) and the two-call runs (15 Sep)
  were executed on an adjacent development line of the same repository. Its agents differ
  from this branch in category normalisation, JSON extraction and confidence forwarding,
  none of which changes the metric the supplement reports (binary FR/NFR accuracy; no
  classification fallbacks occurred in these runs). Only the EN baseline and pipeline
  runs were re-executed on this branch.
- The English runs of the supplement for baseline and pipeline were **re-executed** on
  10 Oct 2026 with this branch's agents. An earlier execution (20 Jun 2026) had sent the
  Portuguese text to the model in the English condition and is **not** included or used.
  A unit test (`tests/unit/test_requirement_language_input.py`) now fails if any agent
  sends Portuguese text in an English condition.
- `manifest.json` records `"git_commit": "Anonymous"` for all runs.

## 3. Environment

- Python >= 3.11; dependencies in `pyproject.toml` (`pip install -e ".[dev]"`).
- Models served locally with Ollama: `qwen2.5:7b`, `llama3.1:8b`, `mistral:7b`
  (Q4_K_M). The paper reports Ollama 0.24.0; the October 2026 repeat and the
  supplement used 0.35.1.
- The `gpt-4.1-mini` supplement uses an Azure AI Foundry deployment, configured through
  the `FOUNDRY_ENDPOINT` and `FOUNDRY_API_KEY` environment variables (never committed).
- Decoding: temperature 0.0. The seed (42) only fixes the dataset sample; it is **not**
  forwarded to the model server.

### Dataset (not included)

`datasets/data/promise_nfr/promise_nfr_pt.csv` is excluded by `.gitignore`. It holds the
625 PROMISE NFR+ requirements with the original English text (`RequirementText`) and a
machine translation (`RequirementText_PT`), and is the file hashed in every manifest:

```
MD5 ea0ec6f6b7c45225819aebbae40b8cff
```

## 4. Running the experiments

From the repository root (Ollama running for the local models):

```bash
python -m experiments.run_grid_full --resume          # 12 baseline/pipeline conditions
python -m experiments.run_ablation_grid --resume      # 6 two-call conditions
python -m experiments.run_repeat_grid --resume        # repeat of the 12 pipeline/two-call conditions
python -m experiments.run_foundry_grid --langs en     # gpt-4.1-mini baseline/pipeline (needs Azure credentials)
```

Each condition takes roughly 50-70 minutes on an RTX 3050 (6 GB).

## 5. Reproducing the tables

All scripts read `experiments/results/` and are run from the repository root.

| Paper element | Script | Output |
|---|---|---|
| RQ1 and language-effect tests (exclusion policy) | `compute_stats.py` | `statistical_results.json` |
| Paired McNemar for RQ1 and language effect | `compute_mcnemar_stats.py` | `statistical_results_mcnemar.json` |
| Same analyses retaining parse-failure fallbacks | `compute_parse_failure_stats.py` | `statistical_results_retained.json` |
| Paired Cohen's g (RQ1 and the state effect) | `compute_cohens_g.py` | `cohens_g_paired.json` |
| Fleiss' kappa per stratum and Must rate | `compute_kappa_table.py` | `kappa_table.json` |
| Bootstrap of the kappa increment | `compute_rq3_kappa_bootstrap.py` | `rq3_kappa_bootstrap_<timestamp>.json` |
| Decomposition ablation (delta_p, delta_s), excluded vs. retained | `compute_delta_f1_retained.py` | `delta_f1_retained.json` |
| Pipeline vs. two-call tests | `compute_ablation_stats.py` | `ablation_<dir>/ablation_stats_<timestamp>.json` |
| Baseline vs. two-call, direct (decomposition alone) | `compute_baseline_vs_two_call_stats.py` | `baseline_vs_two_call.json` |
| Structured-output compliance proxy | `compute_nfr_category_compliance.py` | `nfr_category_compliance.json` |
| NFR subcategory F1 | `compute_subcategory_errorprop.py` | `subcategory_errorprop.json` |
| Run-to-run variation of kappa | `compute_repeat_kappa_variance.py` | `repeat_kappa_variance.json` |
| `gpt-4.1-mini` supplement | `compute_foundry_supplement.py` | `foundry_supplement.json` |

The committed JSON files are the outputs the paper's numbers come from.

## 6. Known limitations of the artifact

- **Not bit-reproducible.** Decoding at temperature 0 is not deterministic on Ollama. In
  the repeated execution, macro F1 of an identical condition varied by up to about one
  point and kappa by up to about 0.03 (`repeat_kappa_variance.json`). Re-running a
  condition reproduces the conclusions, not the exact digits.
- **One execution per condition** for the main grid and the ablation. Only the pipeline
  and two-call conditions were executed twice.
- **Per-call traces are largely absent.** The runner of this version did not write
  per-requirement traces for most runs. Only the two-call `gpt-4.1-mini` supplement runs
  have traces (`experiments/traces/`). Latency is available per run in `manifest.json`.
- **Confidence-aware routing is wired but almost never fires.** The prioritiser receives
  the classifier's confidence and asks for conservative labels below 0.70; this happened
  for 3 of 3,750 pipeline predictions (all parse fallbacks). The two-call baseline fixes
  the confidence at 1.0.
- **Response parsing is regex-based and duplicated across agents.** Malformed JSON
  becomes a deterministic fallback (`FUNCTIONAL` with confidence zero for
  classification, `COULD_HAVE` with score 0.5 for prioritisation). These fallbacks are
  excluded from headline metrics and retained in the sensitivity analyses.
- **Prioritisation fallbacks in the supplement.** With `gpt-4.1-mini` in English, 32
  pipeline and 37 two-call prioritisations fell back because the model emitted a
  trailing comma in its JSON. This does not affect classification accuracy.
- **Translation.** The Portuguese text is a machine translation of the English
  original (back-translation BERTScore F1 mean 0.9646).
