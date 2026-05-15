from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from domain.models import PipelineState

Item = TypeVar("Item")
Output = TypeVar("Output")


class BaseAgent(ABC, Generic[Item, Output]):
    """Abstract base for all MAS4RE agents.

    Generic over:
        Item:   the input element processed per call
                (e.g., Requirement, ClassifiedRequirement).
        Output: the structured result produced per call
                (e.g., ClassifiedRequirement, PrioritizedRequirement).
    """

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState: ...

    @abstractmethod
    def _process_single(self, item: Item) -> Output: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"