# MAS4RE — Multi-Agent System for Requirements Engineering

An empirical study comparing a multi-agent LLM pipeline against single-agent baselines across three Requirements Engineering tasks: **classification**, and **prioritization** of software requirements.

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

Three architectures are evaluated side-by-side:

| Architecture | Description |
|---|---|
| `baseline` | Single LLM call — classification + prioritization in one JSON response |
| `pipeline` | Two specialized agents via LangGraph `StateGraph` with typed state handoff |
| `two_call` | Two sequential LLM calls using the same specialized prompts, but without typed state |

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

Download [PROMISE NFR](http://promise.site.uottawa.ca/SERepository/) and place the files at:

```
datasets/data/promise_nfr/promise_nfr_en.csv   # English original
datasets/data/promise_nfr/promise_nfr_pt.csv   # Portuguese (machine-translated)
```

---

## Running Experiments

### Full grid (18 conditions: 3 models × 3 architectures × 2 languages)

```bash
python scripts/run_grid.py                     # full dataset (n=625)
python scripts/run_grid.py --n 50              # sample of 50 requirements
python scripts/run_grid.py --seed 42           # fixed seed for reproducibility
python scripts/run_grid.py --dry-run           # print conditions without executing
python scripts/run_grid.py --resume            # skip conditions already completed (status=ok)
```

`--resume` skips conditions with `status=ok` in the existing summary CSV, matched by `(strategy, model, lang)`. Conditions with `status=error` are always retried.

### Individual runs

```bash
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
manifest.json   — frozen parameters (model, seed, temperature, dataset hash, git commit)
results.json    — classification metrics (F1, accuracy, MCC) + full predictions
```

And a trace file per requirement:

```
experiments/traces/{run_id}.jsonl   — per-requirement latency + parse outcome
```

A grid summary is appended incrementally:

```
experiments/results/grid_summary.csv
```

### Statistical analysis

After collecting all runs:

```bash
python experiments/compute_stats.py            # RQ1 (Wilcoxon), RQ2 (Mann-Whitney), Fleiss κ
python experiments/compute_ablation_stats.py   # RQ3 ablation (Δ_p and Δ_s)
```

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
| `claude*` | Anthropic | `claude-haiku-4-5` |
| `llama*`, `qwen*`, `mixtral*`, `gemma*` | Groq | `llama-3.3-70b-versatile` |

Switch providers by changing `CLASSIFIER_MODEL` / `PRIORITIZER_MODEL` in `.env` — no code changes required.

---

## Project Structure

```
mas4re/
├── agents/          # BaseAgent, ClassificationAgent, PrioritizationAgent, BaselineAgent
├── config/          # Pydantic settings (loaded from .env)
├── datasets/        # DatasetAdapter ABC + PROMISE adapter
├── domain/          # Pydantic models (Requirement, PipelineState) + enums
├── evaluation/      # Metrics (F1, MCC, Kendall τ, MoSCoW accuracy) + TraceWriter
├── experiments/     # Runner, OrchestrationStrategy, statistical analysis scripts
├── llm/             # LLM factory (Anthropic / Groq / Ollama)
├── pipeline/        # LangGraph StateGraph + node definitions
├── prompts/v1/      # Versioned prompt builders (classification, prioritization, baseline)
├── scripts/         # Entry-point scripts (run_grid.py, run_mas_pipeline.py, run_baseline.py)
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
