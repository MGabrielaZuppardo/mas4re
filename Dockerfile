# =============================================================================
# MAS4RE — Multi-Agent System for Requirements Engineering
# Imagem para reprodutibilidade dos experimentos
# =============================================================================

# ── Stage 1: dependências ────────────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# Dependências de sistema para scipy/numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# Instala dependências de produção + dev (pytest necessário para experimentos)
RUN pip install --no-cache-dir . ".[dev]"

# ── Stage 2: aplicação ──────────────────────────────────────────────────────
FROM python:3.11-slim AS app

WORKDIR /app

# Copia pacotes instalados do stage anterior
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copia código-fonte (respeitando .dockerignore)
COPY . .

# Variáveis de ambiente padrão
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Healthcheck — valida que o pacote importa corretamente
HEALTHCHECK --interval=30s --timeout=5s --retries=2 \
    CMD python -c "from agents.classifier import ClassificationAgent" || exit 1

# Entrypoint padrão: shell interativo (sobrescrito pelo compose/make)
ENTRYPOINT ["python", "-m"]
CMD ["pytest", "tests/unit", "-v"]