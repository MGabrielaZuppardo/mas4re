from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def run(self, state: Any) -> Any: ...

    @abstractmethod
    def _process_single(self, item: Any) -> Any: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
