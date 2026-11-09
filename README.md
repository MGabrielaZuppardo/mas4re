# MAS4RE — Multi-Agent System for Requirements Engineering

An empirical study of how **coordination protocols** shape the behaviour and results of an LLM-based multi-agent system in Requirements Engineering. Two specialised agents (a classifier and a prioritizer, each an autonomous ReAct-style agent) are coordinated by one of three protocols, P0 (fixed), P2 (peers) or P3 (mediated); a non-agentic workflow is kept as an external architectural control. The tasks are FR/NFR classification and MoSCoW priority categorisation on the PrioReSpa benchmark, whose labels are the benchmark's automatic reference criteria, not independent human truth.

The pipeline runs fully locally via [Ollama](https://ollama.com) — no external API calls required during experiments.

---

## Architecture

```
[Input: requirement text]
        ↓
  ClassificationAgent     ← labels FR / NFR + subcategory (11 PROMISE categories)
        ↓
  PipelineState           ← typed Pydantic state shared across agents
        ↓
  PrioritizationAgent     ← assigns MoSCoW priority + confidence-aware routing
        ↓
[Output: structured backlog + trace JSONL]
```

The strategies below are evaluated side-by-side. The dissertation's single experimental factor is the **coordination protocol** of the autonomous agents (`pipeline_agentic*`, ADR-013); `pipeline` is the workflow control.

| Architecture | Description |
|---|---|
| `baseline` | Single LLM call — classification + prioritization in one JSON response |
| `pipeline` | Two specialized agents via LangGraph `StateGraph` with typed state handoff |
| `two_call` | Two sequential LLM calls using the same specialized prompts, but without typed state |
| `pipeline_agentic` (P0, fixed) | Same two roles as `pipeline`, but each is an autonomous agent (ReAct loop): the model chooses its tools and when to finish, and the prioritizer learns about the classification only through a typed message (ADR-012, `docs/agency_conformance.md`). One-way handoff: the prioritizer has no way to ask back |
| `pipeline_agentic_p2` (P2, peers) | P0 plus one round of dialogue: the prioritizer may ask the classifier to reconsider, the classifier agrees or refuses, and the prioritizer decides again (ADR-013) |
| `pipeline_agentic_p3` (P3, mediated) | P0 plus a mediator (environment code, not an agent) that reads the finished backlog, sends typed requests with evidence (critical NFR with a low priority, Must inflation above 60%), and the prioritizer agrees or refuses each one (ADR-013) |

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) (local inference) **or** Anthropic / Groq API keys
- Docker + Docker Compose (optional, for containerized runs)
- NVIDIA GPU recommended (RTX 3050 6 GB or equivalent); CPU-only is slow but functional

---

## Setup

### 1. Clone and install

```bash
git clone <repository-url>
cd mas4re
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# LLM providers (leave blank if using Ollama only)
ANTHROPIC_API_KEY=
GROQ_API_KEY=

# Ollama endpoint (default: local)
OLLAMA_BASE_URL=http://localhost:11434

# Default models for each agent role
CLASSIFIER_MODEL=ollama/qwen2.5:7b
PRIORITIZER_MODEL=ollama/llama3.1:8b

# Experiment parameters
AGENT_TEMPERATURE=0.0
MAX_WORKERS=3
MAX_RETRIES=3

# Dataset paths
PROMISE_DATASET_PATH=datasets/data/promise_nfr/promise_nfr_pt.csv
```

### 3. Pull models (Ollama)

```bash
make pull-models
# or manually:
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull mistral:7b
```

### 4. Place the dataset

**PrioReSpa** (the dissertation's dataset: 401 requirements in Spanish with type, PROMISE category and an automatic MoSCoW reference label) is versioned at `datasets/data/priorespa/PrioReSpa.csv`. The Portuguese version (`PrioReSpa_pt.csv`) is produced and checked by two scripts, in batches of at most 100 items (each command stays under 10 minutes and resumes from a checkpoint):

```bash
python scripts/translate_priorespa.py --model ollama/qwen2.5:7b --limit 25   # first batch
python scripts/validate_translation.py --pair priorespa                       # back-translation + BERTScore
python scripts/translate_priorespa.py                                         # the rest, repeat until done
```

[PROMISE NFR](http://promise.site.uottawa.ca/SERepository/) (no MoSCoW gold) goes at:

```
datasets/data/promise_nfr/promise_nfr_en.csv   # English original
datasets/data/promise_nfr/promise_nfr_pt.csv   # Portuguese (machine-translated)
```

---

## Running Experiments

### Individual runs

```bash
# Coordination protocols on PrioReSpa — primary cell (data ES, prompts ES), 10-item smoke
python -m cli.main run --strategy pipeline_agentic    --dataset-name priorespa --data-lang es --lang es --n 10
python -m cli.main run --strategy pipeline_agentic_p2 --dataset-name priorespa --data-lang es --lang es --n 10
python -m cli.main run --strategy pipeline_agentic_p3 --dataset-name priorespa --data-lang es --lang es --n 10
python -m cli.main run --strategy pipeline            --dataset-name priorespa --data-lang es --lang es --n 10

# MAS pipeline — 50 samples, Portuguese, qwen2.5:7b
python scripts/run_mas_pipeline.py --n 50 --seed 42 --clf-model ollama/qwen2.5:7b

# Full dataset, English, Claude Haiku
python scripts/run_mas_pipeline.py --full --clf-model claude-haiku-4-5

# Baseline agent
python scripts/run_baseline.py --n 50 --seed 42
python scripts/run_baseline.py --full --model claude-haiku-4-5 --lang en --no-taxonomy
```

### Output artifacts

Each run produces files under `experiments/results/{run_id}/`:

```
manifest.json   — frozen parameters (model, seed, temperature, dataset name and hash, data and
                  prompt language, git commit, information_regime, coordination protocol)
results.json    — classification metrics (F1, accuracy, MCC) + full predictions
backlog.csv     — classification + priority of every processed requirement, best first
failed.csv      — requirements the agents could not process, with the recorded reason
                  (only written when there is at least one)
actions.jsonl   — one episode per agent and requirement (goal, beliefs, the actions the agent
                  chose and the ones on offer, tokens); only for `pipeline_agentic*`
messages.jsonl  — every typed message exchanged (the classifier's report, the peer or mediator
                  requests with their evidence, the replies); only when there was any
```

Switching the experiment's model is one edit: the `model:` field of `experiments/configs/confirmatory.yml`
(`<provider>/<name>`, e.g. `ollama/qwen2.5:7b` or `foundry/gpt-4.1-mini`; providers are registered in
`llm/factory.py` with `@provider`, so a new one needs no other change). Check a model before a long run,
then run each arm exactly as frozen in the spec (`run` also checks the model first; `--no-preflight` skips it):

```bash
python -m cli.main check-model <provider>/<name>
python -m cli.main run --arm p0        # p0 | p0-repeat | p2 | p3 | control
```

A run also writes `decisions.jsonl` (the first-pass and final decision of every requirement) and, in
`metrics`, the protocol `conformance`, `interaction_failures` and the cost by phase. The analysis
reports an `architecture_claim` per arm (ADR-015), and any run can be re-checked from its logs alone:

```bash
python -m experiments.replay_metrics <run_dir>
```

Exploratory studies (ADR-016) are kept apart from the confirmatory design: they write to
`experiments/exploratory/`, their manifests say `exploratory`, and they never enter `analyze_protocols`.

```bash
python -m cli.main explore --list
python -m cli.main explore --variant x2-mediator-active --arm p3
python -m cli.main explore --variant x4-llama31 --arm p0
python -m experiments.analyze_exploratory --warmup
python -m experiments.analyze_exploratory --variant x2-mediator-active
```

Each run gets its own folder, `<strategy>_<models>_<data-lang>-<prompt-lang>_n<N>[_<tag>]_<YYYYMMDD-HHMMSS>`
(the local start time; starting the same run twice in the same second fails instead of mixing files).
`manifest.json` and `results.json` also record `started_at`/`executed_at` and `finished_at` (ISO 8601 with
the time zone), and the trace file carries the same name.
`--tag` labels a run in the folder name and in `manifest.json`; the second P0 run (the noise floor)
must be run with `--tag p0-repeat`.

Confirmatory analysis of the three protocols (ADR-014; needs the frozen design and the tagged second
P0 run). It finds the confirmatory run of each arm in `--results-dir` (default `experiments/results`)
and refuses to guess when an arm is missing or ambiguous; explicit `--p0/--p2/--p3/--p0-repeat`
paths take precedence:

```bash
python -m experiments.analyze_protocols
```

`backlog.csv` is UTF-8 with BOM and `;`-separated, so it opens correctly in Excel with Portuguese
locale. One row per requirement: type, NFR category, confidence and justification from the
classifier; priority, score and justification from the prioritizer; and, when the dataset has a
gold label, `reference_type`, `reference_category` and `type_correct`. `failed.csv` adds the stage, mode and
evidence from `failure_detections`, plus every detection recorded for that requirement.

```bash
python -m cli.main run --strategy pipeline --n 50 --backlog-delimiter "," --no-reference-labels
python -m cli.main backlog experiments/results/<run_id>          # rebuild it for a finished run
```

`pipeline_agentic*` runs take `--max-steps` (step budget of each episode, default 5). Their
`results.json` carries an `agent_activity` block per agent: valid and invalid actions, tools
called, episodes ended by budget instead of by the agent, entropy of the action sequences and
tokens; the `manifest.json` records each agent's budget and tools. `interaction` counts the
requests, agreements and refusals between agents, and `mediator` (P3) the findings and the Must
share before and after. On a dataset with MoSCoW reference labels, `prioritization` reports accuracy
next to the majority-class baseline, weighted kappa, rank correlations, per-class precision/recall,
the confusion matrix and coverage.

Languages: `--data-lang` picks the requirement text (`es` or `pt`) and `--lang` the prompts
(`pt`, `en` or `es`). ES/ES is the primary cell; ES/PT and PT/PT are sensitivity checks, and
protocols are only compared inside one cell.

`backlog` rebuilds the CSV from a finished run's `results.json` (options `--delimiter`,
`--no-reference-labels`, `--out`). The requirements the agents could not process are not stored there,
so that command never writes `failed.csv`.

Each run also writes a trace file per requirement:

```
experiments/traces/{run_id}.jsonl   — per-requirement latency + parse outcome
```

### Analysis and diagnostics

Every script below is run from the repository root, writes a JSON with a manifest (git commit,
dataset hash, library versions, parameters and `information_regime`) to
`experiments/results/analysis/`, and can be re-run to reproduce the reported numbers.

Historical analysis, no LLM calls (reads the archived full-dataset runs, `--generation historical`):

```bash
python -m experiments.analyze_error_signals        # which signals predict F/NF errors
python -m experiments.analyze_coordination_power   # cross_check conflicts and Must share per run
python -m experiments.analyze_verifier_oracle      # classical verifier (hybrid, exploratory)
```

Agency and coordination, on real runs of one language cell (`docs/agency_conformance.md`, ADR-013):

```bash
python -m experiments.analyze_agency \
    --agentic experiments/results/<p0_run> --agentic experiments/results/<p2_run> \
    --agentic experiments/results/<p3_run> --workflow experiments/results/<pipeline_run>
```

It reports C1/C2/C4 from the logs, C5 (does an observation change the next action) as an
association test, and C7 (agents against the workflow on the same items, paired by id) with cost and
coverage beside the quality figures. It refuses to mix language cells.

Diagnostics that call the LLM (keep other Ollama work idle while they run):

```bash
python -m experiments.diagnose_classifier_variants   # pre-ADR-011 / current / no critique / no memory
python -m experiments.diagnose_prioritizer_variants  # Must-Have inflation by component
python -m experiments.diagnose_retry_determinism     # does a blind retry reproduce pass 1?
```

### Result generations

`experiments/results/` is for **new** runs (agents from ADR-011 onward). Everything produced before
ADR-011 and the critique fix lives in `experiments/results/archive_pre_adr011_20260920/` (see its
`ARCHIVE.md`). The scripts read it through `experiments/paths.py` and take
`--generation historical|active`, so runs from different generations are never analysed together
by accident. The statistics and grid scripts of the SBQS paper are not on this branch: they remain
on `develop` and `artifact/sbqs-camera-ready`.

`information_regime` is `zero_shot` for everything except `analyze_verifier_oracle`, which trains
on labelled data and is stamped `hybrid_exploratory` so it is never mixed with zero-shot results.

---

## Docker

```bash
# Unit tests only
make test

# App + Ollama (downloads models on first run)
make up-local
# or:
docker compose --profile local up

# Jupyter Lab at http://localhost:8888
docker compose --profile notebook up

# Full stack: app + Ollama + Jupyter
docker compose --profile local,notebook up

# Run specific tests inside the container
docker compose run --rm app pytest tests/unit -v
```

---

## Development

```bash
make lint          # ruff check + format
make lint-fix      # auto-fix lint issues
make typecheck     # mypy over agents config domain evaluation llm prompts
make test-local    # pytest tests/unit (local venv, no Docker)
make test-integ    # integration tests (requires Ollama running)
```

Pre-commit hooks run on every commit: ruff lint + format, secret detection, YAML/TOML validation. Direct commits to `main` are blocked.

```bash
pip install pre-commit
pre-commit install
```

---

## Supported LLM Providers

| Model prefix | Provider | Example |
|---|---|---|
| `ollama/<name>` | Ollama (local) | `ollama/qwen2.5:7b` |
| `deepseek/<model>` | DeepSeek API (OpenAI-compatible; needs `DEEPSEEK_API_KEY`) | `deepseek/deepseek-chat` |
| `claude*` | Anthropic | `claude-haiku-4-5` |
| `llama*`, `qwen*`, `mixtral*`, `gemma*` | Groq | `llama-3.3-70b-versatile` |

Switch providers by changing `CLASSIFIER_MODEL` / `PRIORITIZER_MODEL` in `.env` — no code changes required.

---

## Project Structure

```
mas4re/
├── agents/          # BaseAgent, ClassificationAgent, PrioritizationAgent, BaselineAgent
├── config/          # Pydantic settings (loaded from .env)
├── datasets/        # DatasetAdapter ABC, PROMISE and PrioReSpa adapters, registry, ES→PT translation
├── domain/          # Pydantic models (Requirement, PipelineState) + enums
├── evaluation/      # Metrics (F1, MCC, Kendall τ, MoSCoW accuracy) + TraceWriter
├── experiments/     # Runner, OrchestrationStrategy, statistical analysis scripts
├── llm/             # LLM factory (Anthropic / Groq / Ollama)
├── pipeline/        # LangGraph StateGraph + nodes; protocols.py (P0/P2/P3) and mediation.py
├── prompts/         # v1 (workflow) and v2 (autonomous agents, mediator) prompt builders, PT/EN/ES
├── scripts/         # Entry-point scripts (run_mas_pipeline.py, run_baseline.py)
└── tests/           # Unit tests (pytest)
```

---

## Metrics

| Task | Metrics |
|---|---|
| Classification (FR/NFR) | Macro F1, Accuracy, MCC |
| NFR subcategory | F1 per PROMISE category (n < 30 excluded from primary conclusions) |
| Prioritization | MoSCoW accuracy, must-label rate, Fleiss κ (inter-model consistency) |
| Statistical tests | Wilcoxon signed-rank (RQ1), Mann-Whitney U (RQ2), Bonferroni α′ = 0.0083 over 6 comparisons |
| Effect size | Cohen's h (primary interpretive criterion) |

---

## License

MIT
