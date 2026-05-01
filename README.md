# MAS4RE — Multi-Agent System for Requirements Engineering

> Dissertação de Mestrado — Centro de Informática, Universidade Federal de Pernambuco (CIn/UFPE)

Pipeline multi-agente baseado em LLMs para execução integrada de **elicitação**, **classificação** e **priorização** de requisitos de software, com protocolo de avaliação automatizado e replicável.

---

## Motivação

A Engenharia de Requisitos define o que o software deve ser. Erros nessa fase se propagam por todo o ciclo de desenvolvimento e custam caro. Embora LLMs já demonstrem capacidade em etapas isoladas de RE, nenhum estudo propõe ou avalia um **pipeline integrado** cobrindo as três etapas principais. Este projeto endereça essa lacuna.

---

## Pergunta de Pesquisa

> Em que medida um pipeline multi-agente baseado em LLMs é capaz de executar de forma integrada as etapas de elicitação, classificação e priorização de requisitos de software, e qual a qualidade do output gerado em comparação a abordagens de agente único, mensurada por métricas automáticas e datasets anotados pela comunidade?

### Sub-questões

| # | Questão |
|---|---------|
| SQ1 | Qual a qualidade do output de cada agente individualmente (elicitador, classificador, priorizador), medida contra datasets anotados como PROMISE e NFRIC? |
| SQ2 | O pipeline multi-agente produz outputs de maior qualidade e consistência do que um agente único executando as mesmas tarefas em sequência? |
| SQ3 | Em quais condições o pipeline falha — requisitos ambíguos, conflitos entre agentes, alucinações — e como esses padrões se distribuem entre as etapas? |

---

## Arquitetura

```
Entrada (descrição do sistema)
        │
        ▼
┌───────────────┐
│  Elicitor     │  Gera requisitos atômicos e consistentes
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Classifier   │  Classifica FR/NFR com taxonomia PROMISE NFR+
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Prioritizer  │  Prioriza via MoSCoW com score numérico e ranking
└───────┬───────┘
        │
        ▼
  PipelineState (saída estruturada)
```

---

## Tecnologias

| Categoria | Tecnologia |
|-----------|-----------|
| Linguagem | Python 3.11 |
| Orquestração | LangGraph |
| LLMs Locais | Ollama (qwen2.5:7b, llama3.1:8b, phi3.5:3.8b) |
| LLMs Cloud | Anthropic Claude, Groq |
| Validação | Pydantic v2 |
| Containerização | Docker + WSL2 |
| CI | GitHub Actions |
| Testes | Pytest + pytest-cov |
| Lint/Format | Ruff + Mypy |

---

## Estrutura do Projeto

```
mas4re/
├── agents/
│   ├── base.py              # BaseAgent
│   ├── elicitor.py          # Agente de elicitação
│   ├── classifier.py        # Agente de classificação FR/NFR
│   └── prioritizer.py       # Agente de priorização MoSCoW
├── config/                  # Configurações e settings
├── datasets/
│   └── data/                # Datasets PROMISE NFR+ e NFRIC (não versionados)
├── domain/
│   ├── enums.py             # RequirementType, NFRCategory, MoSCoWPriority
│   └── models.py            # Pydantic models (PipelineState, etc.)
├── evaluation/              # Métricas: precisão, recall, F1, Kendall τ
├── experiments/
│   ├── results/             # Resultados dos experimentos (não versionados)
│   └── logs/                # Logs de execução
├── llm/
│   └── factory.py           # build_llm() — factory para todos os providers
├── prompts/
│   └── v1/                  # Templates de prompt por agente
├── tests/
│   ├── unit/                # Testes unitários (sem LLM)
│   └── integration/         # Testes de integração (requer Ollama)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Python 3.11+](https://www.python.org/) (para desenvolvimento local)
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes)

---

## Instalação e Uso

### Com Docker (recomendado)

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/mas4re.git
cd mas4re

# Copiar e configurar variáveis de ambiente
cp .env.example .env

# Rodar testes unitários
docker compose run app pytest tests/unit -v

# Rodar com Ollama local (puxa modelos automaticamente)
docker compose --profile local up

# Rodar testes de integração (requer Ollama rodando)
docker compose run app pytest tests/integration -v
```

### Desenvolvimento Local

```bash
# Instalar dependências
uv pip install -e ".[dev]"

# Rodar testes unitários
pytest tests/unit -v --cov=agents --cov-report=term-missing

# Lint e formatação
ruff check .
ruff format .
mypy agents config domain evaluation llm prompts
```

---

## Variáveis de Ambiente

Crie um arquivo `.env` baseado no `.env.example`:

```env
# Provider padrão: ollama | anthropic | groq
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434

# Modelos por agente
ELICITOR_MODEL=ollama/qwen2.5:7b
CLASSIFIER_MODEL=ollama/qwen2.5:7b
PRIORITIZER_MODEL=ollama/llama3.1:8b

# APIs externas (opcional)
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

---

## Modelos Suportados

| Modelo | Provider | Uso |
|--------|----------|-----|
| `qwen2.5:7b` | Ollama (local) | Classificação |
| `llama3.1:8b` | Ollama (local) | Priorização |
| `phi3.5:3.8b` | Ollama (local) | Baseline |
| `claude-3-5-sonnet` | Anthropic | Cloud |
| `llama3-8b-8192` | Groq | Cloud rápido |

---

## Datasets

| Dataset | Descrição | Uso |
|---------|-----------|-----|
| PROMISE NFR+ | 625 requisitos anotados (Cleland-Huang et al., 2007) | Classificação FR/NFR |
| NFRIC | Dataset de categorias NFR | Validação de categorias |

Os datasets devem ser colocados em `datasets/data/` (não versionados no git).

---

## Métricas de Avaliação

- **Classificação:** Precisão, Recall, F1-score (macro/weighted)
- **Priorização:** Kendall τ, Spearman ρ, concordância MoSCoW
- **Pipeline:** Comparação MAS vs. agente único baseline

---

## CI/CD

```
push/PR → Lint & Typecheck (ruff + mypy) → Testes Unitários → Upload cobertura
```

- Lint: `ruff check .`
- Typecheck: `mypy agents config domain evaluation llm prompts`
- Testes: `pytest tests/unit -v --cov`

---

## Contribuições Esperadas

| # | Contribuição |
|---|-------------|
| C1 | Arquitetura de referência reutilizável de pipeline multi-agente para ER |
| C2 | Protocolo de avaliação automatizado e replicável |
| C3 | Benchmark comparativo MAS vs. agente único |
| C4 | Mapeamento de padrões de falha por etapa do pipeline |

---

## Referências

- Zadenoori et al. (2025) — Systematic review LLM4RE (74 estudos)
- Hemmat et al. (2025) — Revisão 2020–2024, Frontiers CS
- He, Treude & Lo (2025) — MAS em SE, ACM TOSEM
- Hey et al. (2020) — Classificação FR/NFR, IEEE RE
- Ronanki et al. (2023) — ChatGPT em elicitação, IEEE RE
- Perini et al. (2013) — Priorização MoSCoW, RE Journal
- Cleland-Huang et al. (2007) — Dataset PROMISE NFR+

---

## Licença

Este projeto é parte de uma dissertação de mestrado no CIn/UFPE. Todos os direitos reservados.

---

<p align="center">
  Desenvolvido no <strong>Centro de Informática — UFPE</strong>
</p>
