from __future__ import annotations

import random
from abc import ABC, abstractmethod

from domain.models import Requirement


class DatasetAdapter(ABC):
    """Interface mínima que todo adapter de dataset deve implementar.

    Garante que qualquer dataset possa ser consumido pelos agentes sem
    que eles precisem conhecer o schema interno da fonte de dados.

    Uso:
    adapter = PromiseAdapter()
    requirements = adapter.load()
    agent=BaselineAgent(model=...,nfr_categories=adapter.nfr_categories)
    """

    @abstractmethod
    def load(self) -> list[Requirement]:
        "Carrega os requisitos do dataset como objetos Requirements"

    def load_sample(self, n: int, seed: int = 42) -> list[Requirement]:
        """Amostra aleatória reprodutível de n requisitos (tudo, se n >= tamanho)."""
        requirements = self.load()
        return random.Random(seed).sample(requirements, min(n, len(requirements)))

    @property
    def nfr_categories(self) -> list[tuple[str, str]] | None:
        """
        Retorna as categorias NFR como lista de (código, descrição)

        Retorna None se o dataset não tiver categorias NFR de maneira estruturada.
        Os agentes usam essa lista para montar o bloco de categorias no seu prompt.
        """
        return None

    @property
    def name(self) -> str:
        """Nome do dataset para loggin e relatórios de experimento"""
        return self.__class__.__name__

    def __repr__(self) -> str:
        n_cats = len(self.nfr_categories) if self.nfr_categories else 0
        return f"{self.name}(nfr_categories={n_cats})"
