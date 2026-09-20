"""Fixtures compartilhadas dos testes de integração.

Estes testes exigem um LLM real e um dataset anotado. Nada é fixo no ``.env``:
modelo e dataset vêm de variáveis de ambiente, e os testes são pulados quando o
Ollama ou o modelo não estão disponíveis.

Variáveis de ambiente:
    INTEGRATION_CLASSIFIER_MODEL / INTEGRATION_PRIORITIZER_MODEL
        Modelos dos agentes (padrão: ollama/qwen2.5:7b e ollama/llama3.1:8b).
    INTEGRATION_DATASETS
        Datasets de ``_DATASETS`` separados por vírgula, ou ``all`` (padrão: promise).
    INTEGRATION_SAMPLE_SIZE
        Requisitos por dataset (padrão: 15).

Para um dataset entrar em ``_DATASETS``, ``load_sample(n, seed)`` deve devolver
``Requirement`` com ``metadata["label_type"]`` em {"F", "NF"} (a mesma chave lida por
``evaluation.metrics.classification``). Requisitos sem essa chave rodam os testes que
não precisam de gabarito e pulam os demais.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from config.settings import settings
from datasets.promise import PromiseAdapter
from domain.models import Requirement

_OLLAMA_PREFIX = "ollama/"
_TAGS_TIMEOUT_SECONDS = 3
_DEFAULT_CLASSIFIER_MODEL = "ollama/qwen2.5:7b"
_DEFAULT_PRIORITIZER_MODEL = "ollama/llama3.1:8b"
_DEFAULT_SAMPLE_SIZE = 15
_SEED = 42
_GROUND_TRUTH_LABELS = {"F", "NF"}


@dataclass(frozen=True)
class IntegrationDataset:
    name: str
    load_sample: Callable[[int, int], list[Requirement]]
    nfr_categories: list[tuple[str, str]] | None = None


def _load_promise_sample(n: int, seed: int) -> list[Requirement]:
    return PromiseAdapter(path=settings.promise_dataset_path).load_sample(n, seed=seed)


_DATASETS: dict[str, IntegrationDataset] = {
    "promise": IntegrationDataset(name="promise", load_sample=_load_promise_sample),
}


def _installed_ollama_models() -> set[str]:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    with urllib.request.urlopen(url, timeout=_TAGS_TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    return {model["name"] for model in payload["models"]}


def _require_available(model: str) -> str:
    if not model.startswith(_OLLAMA_PREFIX):
        return model
    name = model.removeprefix(_OLLAMA_PREFIX)
    try:
        installed = _installed_ollama_models()
    except (OSError, ValueError, KeyError) as e:
        pytest.skip(f"Ollama indisponível em {settings.ollama_base_url}: {e}")
    if (name if ":" in name else f"{name}:latest") not in installed:
        pytest.skip(f"Modelo {name!r} não baixado no Ollama (ollama pull {name}).")
    return model


def _selected_dataset_names() -> list[str]:
    raw = os.environ.get("INTEGRATION_DATASETS", "promise").strip()
    names = list(_DATASETS) if raw == "all" else [n.strip() for n in raw.split(",") if n.strip()]
    unknown = [n for n in names if n not in _DATASETS]
    if unknown:
        raise pytest.UsageError(
            f"INTEGRATION_DATASETS desconhecido(s): {unknown}. Registrados: {list(_DATASETS)}"
        )
    return names


@pytest.fixture(scope="session")
def classifier_model() -> str:
    return _require_available(
        os.environ.get("INTEGRATION_CLASSIFIER_MODEL", _DEFAULT_CLASSIFIER_MODEL)
    )


@pytest.fixture(scope="session")
def prioritizer_model() -> str:
    return _require_available(
        os.environ.get("INTEGRATION_PRIORITIZER_MODEL", _DEFAULT_PRIORITIZER_MODEL)
    )


@pytest.fixture(scope="session", params=_selected_dataset_names())
def dataset(request: pytest.FixtureRequest) -> IntegrationDataset:
    return _DATASETS[request.param]


@pytest.fixture(scope="session")
def dataset_sample(dataset: IntegrationDataset) -> list[Requirement]:
    size = int(os.environ.get("INTEGRATION_SAMPLE_SIZE", _DEFAULT_SAMPLE_SIZE))
    return dataset.load_sample(size, _SEED)


@pytest.fixture(scope="session")
def labeled_sample(
    dataset: IntegrationDataset, dataset_sample: list[Requirement]
) -> list[Requirement]:
    missing = [
        r.id for r in dataset_sample if r.metadata.get("label_type") not in _GROUND_TRUTH_LABELS
    ]
    if missing:
        pytest.skip(
            f"Dataset {dataset.name!r} sem metadata['label_type'] em {len(missing)} requisitos."
        )
    return dataset_sample
