from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents.base import BaseAgent
from domain.enums import Lang, MoSCoWPriority, NFRCategory, RequirementType
from domain.failures import FailureRecord, FailureSeverity

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter
from domain.models import (
    BaselineOutput,
    PipelineState,
    PrioritizedRequirement,
    Requirement,
)
from evaluation.failure_detectors import DetectionContext, default_chain, max_severity
from llm.factory import build_llm
from llm.json_parser import coerce_str, extract_first_json
from prompts.v1.baseline import build_baseline_messages

logger = logging.getLogger(__name__)

_KNOWN_NFR_CATEGORIES = {c.value for c in NFRCategory.nfr_only()}
_CONFIDENCE_UNCERTAIN_THRESHOLD = 0.70


class BaselineAgent(BaseAgent[Requirement, PrioritizedRequirement]):
    """Agente baseline single-shot: classifica e prioriza em uma única chamada de LLM."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        nfr_categories: list[tuple[str, str]] | None = None,
        lang: Lang = Lang.PT,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, trace_writer=trace_writer)
        self._llm = build_llm(model, temperature)
        self._nfr_categories = nfr_categories
        self._lang: Lang = lang
        self._detector_chain = default_chain()
        logger.info(
            "BaselineAgent inicializado | model=%s | lang=%s | nfr_categories=%s",
            model,
            lang.value,
            len(nfr_categories) if nfr_categories else "None",
        )

    def run(self, state: PipelineState) -> PipelineState:
        logger.info("Iniciando baseline | run_id=%s | n=%d", state.run_id, state.n_requirements)
        failed_ids: list[str] = []
        failure_records: list[FailureRecord] = []
        prioritized = self._run_batch(
            state.raw_requirements,
            max_workers=3,
            failed_ids=failed_ids,
            failure_records=failure_records,
        )
        state.prioritized_requirements = prioritized
        state.model_used = self.model
        state.errors.extend(f"baseline_failed:{req_id}" for req_id in failed_ids)
        state.failure_records.extend(failure_records)
        n_fatal = sum(1 for r in failure_records if r.severity is FailureSeverity.FATAL)
        logger.info(
            "Baseline concluído | run_id=%s | processados=%d | dropados=%d | fatal=%d | flagged=%d",
            state.run_id,
            len(prioritized),
            len(failed_ids),
            n_fatal,
            len(failure_records) - n_fatal,
        )
        return state

    def _run_batch(
        self,
        requirements: list[Requirement],
        max_workers: int = 3,
        failed_ids: list[str] | None = None,
        failure_records: list[FailureRecord] | None = None,
    ) -> list[PrioritizedRequirement]:
        results: dict[str, PrioritizedRequirement] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._call_and_trace, "baseline", req): req for req in requirements
            }
            for future in as_completed(futures):
                req = futures[future]
                try:
                    output = future.result()
                except Exception as e:
                    logger.error("Falha ao processar | id=%s | erro=%s", req.id, e)
                    if failed_ids is not None:
                        failed_ids.append(req.id)
                    continue

                ctx = DetectionContext(
                    requirement_id=output.id,
                    stage="baseline",
                    requirement_text=req.text,
                    parsed_ok=not output.parse_failed,
                    confidence=output.confidence,
                    predicted_category=output.nfr_category,
                    known_categories=_KNOWN_NFR_CATEGORIES,
                    justification=output.justification,
                    confidence_threshold=_CONFIDENCE_UNCERTAIN_THRESHOLD,
                )
                records = self._detector_chain.run(ctx)
                if failure_records is not None:
                    failure_records.extend(records)
                if max_severity(records) is FailureSeverity.FATAL:
                    continue
                results[req.id] = output

        ordered = [results[r.id] for r in requirements if r.id in results]
        ordered.sort(key=lambda r: r.priority_score or 0.0, reverse=True)
        for rank, req in enumerate(ordered, start=1):
            req.priority_rank = rank
        return ordered

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=30, min=30, max=240),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _process_single(self, requirement: Requirement) -> PrioritizedRequirement:
        messages = build_baseline_messages(
            requirement_text=requirement.text,
            lang=self._lang,
            nfr_categories=self._nfr_categories,
        )
        response = self._llm.invoke(messages)
        output = self._parse_response(str(response.content), requirement.id)
        return PrioritizedRequirement.from_baseline(requirement, output)

    def _parse_response(self, content: str, req_id: str) -> BaselineOutput:
        try:
            data = json.loads(extract_first_json(content))
            req_type = RequirementType(data["requirement_type"])
            priority = MoSCoWPriority(data["priority"])
            return BaselineOutput(
                requirement_id=req_id,
                requirement_type=req_type,
                nfr_category=data.get("nfr_category"),
                confidence=float(data.get("confidence", 0.5)),
                classification_justification=coerce_str(
                    data.get("classification_justification", "")
                ),
                priority=priority,
                priority_score=float(data.get("priority_score", priority.score)),
                priority_rank=int(data.get("priority_rank", 1)),
                priority_justification=coerce_str(data.get("priority_justification", "")),
            )
        except Exception as e:
            logger.error(
                "Parse falhou | req_id=%s | erro=%s | conteúdo=%r",
                req_id,
                e,
                content[:200],
            )
            return BaselineOutput(
                requirement_id=req_id,
                requirement_type=RequirementType.FUNCTIONAL,
                nfr_category=None,
                confidence=0.0,
                classification_justification=f"Parse falhou: {e}",
                priority=MoSCoWPriority.COULD_HAVE,
                priority_score=0.5,
                priority_rank=1,
                priority_justification=f"Parse falhou: {e}",
                parse_failed=True,
            )
