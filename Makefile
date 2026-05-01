# =============================================================================
# MAS4RE — Makefile
# Comandos para reprodutibilidade dos experimentos
# =============================================================================

.PHONY: help build test test-unit test-integ lint typecheck \
        up up-local down pull-models clean

# ── Variáveis ────────────────────────────────────────────────────────────────
IMAGE       := mas4re-app
COMPOSE     := docker compose
COMPOSE_LOCAL := $(COMPOSE) --profile local
PYTEST      := python -m pytest
VENV_PY     := .venv/Scripts/python

help: ## Exibe esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

build: ## Constrói a imagem Docker
	$(COMPOSE) build app

test: ## Roda todos os testes unitários no container
	$(COMPOSE) run --rm app pytest tests/unit -v

test-unit: ## Alias para test
	$(COMPOSE) run --rm app pytest tests/unit -v

test-integ: ## Roda testes de integração (requer Ollama)
	$(COMPOSE_LOCAL) run --rm app pytest tests/integration -v -s

test-local: ## Roda testes unitários localmente
	$(VENV_PY) -m pytest tests/unit -v

test-integ-local: ## Roda testes de integração localmente (requer Ollama no host)
	$(VENV_PY) -m pytest tests/integration -v -s

# ── Qualidade de código ─────────────────────────────────────────────────────
lint: ## Roda ruff (lint + format check)
	$(VENV_PY) -m ruff check .
	$(VENV_PY) -m ruff format --check .

lint-fix: ## Corrige problemas de lint automaticamente
	$(VENV_PY) -m ruff check --fix .
	$(VENV_PY) -m ruff format .

typecheck: ## Roda mypy
	$(VENV_PY) -m mypy agents config domain evaluation llm prompts

up: ## Sobe apenas o app (Ollama externo)
	$(COMPOSE) up app

up-local: ## Sobe app + Ollama local (download de modelos na 1ª vez)
	$(COMPOSE_LOCAL) up

down: ## Para todos os serviços
	$(COMPOSE) down

pull-models: ## Puxa modelos Ollama necessários (requer Ollama local rodando)
	ollama pull qwen2.5:7b
	ollama pull llama3.1:8b
	ollama pull phi3.5:3.8b


clean: ## Remove containers, imagens e volumes do projeto
	$(COMPOSE) down --rmi local --volumes --remove-orphans
	docker image prune -f --filter "label=project=mas4re"