# =============================================================================
# MAS4RE — Multi-Agent System for Requirements Engineering
# =============================================================================
# syntax=docker/dockerfile:1

# ── Estágio builder — instala dependências em venv isolada ──────────────────
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
RUN touch README.md

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install \
    numpy>=1.26.0 pandas>=2.0.0 scikit-learn>=1.3.0 scipy>=1.11.0

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    anthropic>=0.40.0 langchain>=0.3.0 langchain-anthropic>=0.3.0 \
    langchain-groq>=0.2.0 langchain-ollama>=0.2.0 langchain-openai>=0.2.0 langgraph>=0.2.0

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    tenacity>=8.0.0 pydantic>=2.0.0 pydantic-settings>=2.6.0 \
    python-dotenv>=1.0.0 typer>=0.12.0 rich>=13.0.0 \
    azure-storage-blob>=12.0.0 \
    pytest>=8.0.0 pytest-cov>=5.0.0 ruff>=0.4.0 mypy>=1.10.0

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    jupyter>=1.0.0 ipykernel>=6.0.0 ipywidgets>=8.0.0 \
    matplotlib>=3.7.0 seaborn>=0.13.0 tqdm>=4.66.0

# ── Estágio runtime — imagem final, sem uv nem cache de build ───────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY --from=builder /opt/venv /opt/venv
COPY . .
RUN chown -R app:app /app

USER app

HEALTHCHECK --interval=30s --timeout=5s --retries=2 \
    CMD python -c "from agents.classifier import ClassificationAgent" || exit 1

ENTRYPOINT ["python", "-m"]
CMD ["pytest", "tests/unit", "-v"]
