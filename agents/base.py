from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from domain.models import PipelineState

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter

Item = TypeVar("Item")
Output = TypeVar("Output")


class BaseAgent(ABC, Generic[Item, Output]):
    """Abstract base for all MAS4RE agents.

    Generic over:
        Item:   the input element processed per call
                (e.g., Requirement, ClassifiedRequirement).
        Output: the structured result produced per call
                (e.g., ClassifiedRequirement, PrioritizedRequirement).

    Pass a TraceWriter instance to capture per-call latency and parse
    outcome in a JSONL file (PR #22 / §5 Artifact Availability).
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._trace = trace_writer

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState: ...

    @abstractmethod
    def _process_single(self, item: Item) -> Output: ...

    def _call_and_trace(self, stage: str, item: Any) -> Any:
        """Wrap _process_single with latency measurement and trace emission.

        Concrete agents submit this method to ThreadPoolExecutor instead of
        _process_single so every call is instrumented uniformly.
        """
        from evaluation.trace_writer import TraceEvent

        t0 = time.monotonic()
        parsed_ok = True
        try:
            return self._process_single(item)
        except Exception:
            parsed_ok = False
            raise
        finally:
            if self._trace is not None:
                self._trace.write(
                    TraceEvent(
                        stage=stage,
                        requirement_id=getattr(item, "id", "unknown"),
                        model=self.model,
                        latency_ms=round((time.monotonic() - t0) * 1000, 2),
                        parsed_ok=parsed_ok,
                        timestamp=datetime.now(UTC),
                    )
                )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
