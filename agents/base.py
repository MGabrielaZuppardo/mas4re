from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents.memory import EpisodicMemory
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

# Warm-up size for agents with an active EpisodicMemory (ADR-011): the first
# _WARMUP_SIZE items of a batch are processed sequentially so memory.add()
# sees them in order, before memory.freeze() unblocks the usual parallel
# phase for the rest. Agents without memory (self._memory is None) are
# unaffected and run fully parallel as before.
_WARMUP_SIZE = 10


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
        # Set by ClassificationAgent/PrioritizationAgent per batch call
        # (ADR-011); None here means "no memory" -- _run_batch stays fully
        # parallel, unchanged for BaselineAgent/TwoCallBaselineAgent and for
        # the streaming graph's process_item_node.
        self._memory: EpisodicMemory | None = None

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

    def _process_and_detect(
        self,
        item: Item,
        stage: str,
    ) -> tuple[str, Output | None, list[FailureRecord]]:
        """Process one item and run SQ3 failure detection on the outcome.

        Pure with respect to shared batch state (no writes to a shared dict
        or list) -- safe to call from a worker thread. The caller (always
        single-threaded at its call site: either the sequential warm-up loop
        or the as_completed loop below) does the results/failure-list
        bookkeeping, matching how _run_batch always worked before ADR-011.
        """
        item_id = item.id  # type: ignore[attr-defined]
        output: Output | None = None
        parsed_ok = True
        try:
            output = self._call_and_trace(stage, item)
        except Exception as e:
            parsed_ok = False
            logger.error("Falha ao processar (%s) | id=%s | erro=%s", stage, item_id, e)
        ctx = self._detection_context(item, output, stage, parsed_ok)
        records = _DETECTOR_CHAIN.run(ctx) if ctx is not None else []
        return item_id, output, records

    def _run_batch(
        self,
        items: list[Item],
        stage: str,
        max_workers: int | None = None,
    ) -> list[Output]:
        """Run _process_single over *items*, preserving input order in the result.

        Shared by every agent's batch entry point (classify_batch,
        prioritize_batch, etc.). Two modes, selected by self._memory:

        - self._memory is None (BaselineAgent/TwoCallBaselineAgent, or the
          streaming graph's process_item_node): fully parallel via
          ThreadPoolExecutor, as before ADR-011.
        - self._memory is set (ClassificationAgent/PrioritizationAgent,
          ADR-011): the first _WARMUP_SIZE items run sequentially so
          memory.add() sees them in order, then memory.freeze() before the
          rest run in parallel -- concurrent reads of a frozen memory never
          race a concurrent write.

        Logs and drops any item whose call raised after retries. Also
        accumulates the SQ3 failure-detection records from every item
        (success or failure) into self._failure_records.
        """
        workers = max_workers if max_workers is not None else settings.max_workers
        results: dict[str, Output] = {}
        self._failure_records = []

        def _record(item_id: str, output: Output | None, records: list[FailureRecord]) -> None:
            if output is not None:
                results[item_id] = output
            self._failure_records.extend(records)

        remaining = items
        if self._memory is not None:
            warmup, remaining = items[:_WARMUP_SIZE], items[_WARMUP_SIZE:]
            for item in warmup:
                _record(*self._process_and_detect(item, stage))
            self._memory.freeze()

        if remaining:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._process_and_detect, item, stage) for item in remaining
                ]
                for future in as_completed(futures):
                    _record(*future.result())

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
