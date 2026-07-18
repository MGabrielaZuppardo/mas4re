from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from domain.failures import FailureRecord
from domain.models import PipelineState, PrioritizedRequirement
from evaluation.failure_detectors import DetectionContext, default_chain

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter

logger = logging.getLogger(__name__)

Item = TypeVar("Item")
Output = TypeVar("Output")

# Shared retry policy for every agent's LLM call (_process_single): capped
# exponential backoff (settings.retry_wait_*), re-raising after the
# configured number of attempts so the batch loop's except-block records
# the failure instead of retrying forever.
llm_retry = retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(
        multiplier=settings.retry_wait_multiplier,
        min=settings.retry_wait_min,
        max=settings.retry_wait_max,
    ),
    stop=stop_after_attempt(settings.max_retries),
    reraise=True,
)

# Stateless v1 detector chain (ADR-003 / SQ3), shared by every agent's
# _run_batch call.
_DETECTOR_CHAIN = default_chain()


def rank_by_priority(
    requirements: list[PrioritizedRequirement],
) -> list[PrioritizedRequirement]:
    """Sort by priority_score descending and assign global priority_rank."""
    requirements.sort(key=lambda r: r.priority_score or 0.0, reverse=True)
    for rank, req in enumerate(requirements, start=1):
        req.priority_rank = rank
    return requirements


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
        self._failure_records: list[FailureRecord] = []

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState: ...

    @abstractmethod
    def _process_single(self, item: Item) -> Output: ...

    def _detection_context(
        self,
        item: Item,
        output: Output | None,
        stage: str,
        parsed_ok: bool,
    ) -> DetectionContext | None:
        """Build the SQ3 failure-detection context for one processed item.

        Default is a no-op (returns None, detection skipped); concrete
        agents override this to map their Output shape onto DetectionContext
        fields. Called for both successful and failed items (output is None
        on failure) so detect_schema can fire even when parsing never
        produced an Output.
        """
        return None

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

    def _run_batch(
        self,
        items: list[Item],
        stage: str,
        max_workers: int | None = None,
    ) -> list[Output]:
        """Run _process_single over *items* in parallel, preserving input order.

        Shared by every agent's batch entry point (classify_batch,
        prioritize_batch, etc.): submits _call_and_trace to a
        ThreadPoolExecutor, collects results keyed by item.id, logs and
        drops any item whose call raised after retries. Also runs the SQ3
        failure-detection chain per item (success or failure) via
        _detection_context, accumulating records in self._failure_records.
        """
        workers = max_workers if max_workers is not None else settings.max_workers
        results: dict[str, Output] = {}
        self._failure_records = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._call_and_trace, stage, item): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                item_id = item.id  # type: ignore[attr-defined]
                output: Output | None = None
                parsed_ok = True
                try:
                    output = future.result()
                    results[item_id] = output
                except Exception as e:
                    parsed_ok = False
                    logger.error("Falha ao processar (%s) | id=%s | erro=%s", stage, item_id, e)
                ctx = self._detection_context(item, output, stage, parsed_ok)
                if ctx is not None:
                    self._failure_records.extend(_DETECTOR_CHAIN.run(ctx))
        return [results[item.id] for item in items if item.id in results]  # type: ignore[attr-defined]

    def _persist_failure_records(self, state: PipelineState) -> None:
        """Append this agent's accumulated SQ3 failure records into shared state.

        Call after a batch call in run(); safe no-op if nothing was flagged.
        Uses "failure_detections" so multiple stages (e.g. classifier then
        prioritizer in the pipeline) accumulate into the same list.
        """
        if not self._failure_records:
            return
        state.metrics.setdefault("failure_detections", []).extend(
            r.model_dump(mode="json") for r in self._failure_records
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
