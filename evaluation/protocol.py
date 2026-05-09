"""
MAS4RE — Protocolo de Avaliação

Define os parâmetros fixos do experimento para garantir reprodutibilidade.
Todos os experimentos devem usar estas constantes.
"""

from __future__ import annotations

# ── Reprodutibilidade ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEMPERATURE = 0.0          # LLM determinístico

# ── Amostragem ────────────────────────────────────────────────────────────────
SAMPLE_SIZES = [50, 100, None]   # None = dataset completo
DEFAULT_SAMPLE = 50              # Usado em testes rápidos

# ── Modelos avaliados ─────────────────────────────────────────────────────────
CLASSIFIER_MODELS = [
    "ollama/qwen2.5:7b",
    "ollama/llama3.1:8b",
    "ollama/phi3.5:3.8b",
]

PRIORITIZER_MODELS = [
    "ollama/llama3.1:8b",
    "ollama/qwen2.5:7b",
    "ollama/phi3.5:3.8b",
]

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = {
    "promise_pt": "datasets/data/promise_nfr/promise_nfr_pt.csv",
    "promise_en": "datasets/data/promise_nfr/promise_nfr_en.csv",
}

# ── Paralelismo ───────────────────────────────────────────────────────────────
MAX_WORKERS = 3

# ── Métricas esperadas ────────────────────────────────────────────────────────
# Thresholds mínimos para considerar o agente aceitável (SQ1)
MIN_F1_MACRO = 0.70
MIN_ACCURACY = 0.75
