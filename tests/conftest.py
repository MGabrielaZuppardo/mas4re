import pytest
from pathlib import Path
from datasets.promise import PromiseAdapter
from config.settings import settings

SAMPLE_N = 15


@pytest.fixture(scope="session")
def promise_adapter():
    return PromiseAdapter(path=settings.promise_dataset_path)


@pytest.fixture(scope="session")
def promise_sample(promise_adapter):
    """15 requisitos reais do PROMISE NFR+ para testes de integração."""
    return promise_adapter.load_sample(n=SAMPLE_N, seed=42)


@pytest.fixture(scope="session")
def promise_full(promise_adapter):
    """Dataset completo — apenas testes pesados."""
    return promise_adapter.load()