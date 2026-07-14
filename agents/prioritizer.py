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
from domain.enums import Lang, MoSCoWPriority
from domain.failures import FailureRecord, FailureSeverity

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter
from domain.models import (
    ClassifiedRequirement,
    PipelineState,
    PrioritizationOutput,
    PrioritizedRequirement,
)
from evaluation.failure_detectors import DetectionContext, default_chain, max_severity
from llm.factory import build_llm
from llm.json_parser import coerce_str, extract_first_json
from prompts.v1.prioritization import build_prioritization_messages

logger = logging.getLogger(__name__)


class PrioritizationAgent(BaseAgent[ClassifiedRequirement, PrioritizedRequirement]):
    """
    Agente de priorização MoSCoW baseado em LLM.

    Prioriza requisitos classificados atribuindo categoria MoSCoW,
    score numérico, ranking e justificativa.

    Features:
    - Retry com backoff exponencial (até 3 tentativas)
    - Execução paralela via ThreadPoolExecutor
    - Saída estruturada validada com Pydantic
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        lang: Lang = Lang.PT,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, trace_writer=trace_writer)
        self._llm = build_llm(model, temperature)
        self._lang = lang
        self._detector_chain = default_chain()
        logger.info("PrioritizationAgent inicializado | model=%s | lang=%s", model, lang.value)

    def run(self, state: PipelineState) -> PipelineState:
        """Prioriza todos os requisitos classificados do estado."""
        logger.info(
            "Iniciando priorização | run_id=%s | n=%d",
            state.run_id,
            len(state.classified_requirements),
        )
        failed_ids: list[str] = []
        failure_records: list[FailureRecord] = []
        prioritized = self.prioritize_batch(
            state.classified_requirements,
            max_workers=3,
            failed_ids=failed_ids,
            failure_records=failure_records,
        )
        state.prioritized_requirements = prioritized
        state.errors.extend(f"prioritize_failed:{req_id}" for req_id in failed_ids)
        state.failure_records.extend(failure_records)
        n_fatal = sum(1 for r in failure_records if r.severity is FailureSeverity.FATAL)
        logger.info(
            "Priorização concluída | run_id=%s | priorizados=%d | "
            "dropados=%d | fatal=%d | flagged=%d",
            state.run_id,
            len(prioritized),
            len(failed_ids),
            n_fatal,
            len(failure_records) - n_fatal,
        )
        return state

    def prioritize_batch(
        self,
        requirements: list[ClassifiedRequirement],
        max_workers: int = 3,
        failed_ids: list[str] | None = None,
        failure_records: list[FailureRecord] | None = None,
    ) -> list[PrioritizedRequirement]:
        """Prioriza requisitos em paralelo e aplica ranking global.

        Runs the ADR-003 DetectorChain on every processed item (schema and
        grounding apply; confidence/category detectors self-no-op since
        PrioritizationOutput has neither signal). Items whose most severe
        FailureRecord is FATAL are excluded from the returned list.

        Args:
            failed_ids: se fornecida, recebe (via append) os ids que
                falharam após esgotar as tentativas de retry — não
                presentes no retorno.
            failure_records: se fornecida, recebe (via extend) todos os
                FailureRecords emitidos pela DetectorChain.
        """
        results: dict[str, PrioritizedRequirement] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._call_and_trace, "prioritize", req): req
                for req in requirements
            }
            for future in as_completed(futures):
                req = futures[future]
                try:
                    output = future.result()
                except Exception as e:
                    logger.error("Falha ao priorizar | id=%s | erro=%s", req.id, e)
                    if failed_ids is not None:
                        failed_ids.append(req.id)
                    continue

                ctx = DetectionContext(
                    requirement_id=output.id,
                    stage="prioritize",
                    requirement_text=req.text,
                    parsed_ok=not output.priority_parse_failed,
                    justification=output.priority_justification,
                )
                records = self._detector_chain.run(ctx)
                if failure_records is not None:
                    failure_records.extend(records)
                if max_severity(records) is FailureSeverity.FATAL:
                    continue
                results[req.id] = output

        # Preserva ordem original
        ordered = [results[r.id] for r in requirements if r.id in results]

        # Aplica ranking global por priority_score decrescente
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
    def _process_single(self, requirement: ClassifiedRequirement) -> PrioritizedRequirement:
        """Prioriza um requisito com retry/backoff."""
        req_text = (
            requirement.text_en
            if self._lang is Lang.EN and requirement.text_en
            else requirement.text
        )
        messages = build_prioritization_messages(
            requirement_text=req_text,
            requirement_type=requirement.requirement_type.value,
            nfr_category=requirement.nfr_category,
            confidence=requirement.confidence,
            lang=self._lang,
        )
        response = self._llm.invoke(messages)
        output = self._parse_response(str(response.content), requirement.id)
        return PrioritizedRequirement.from_classified(requirement, output)

    def _parse_response(self, content: str, req_id: str) -> PrioritizationOutput:
        """Parse do JSON retornado pelo LLM com fallback seguro."""
        try:
            data = json.loads(extract_first_json(content))

            priority = MoSCoWPriority(data["priority"])

            return PrioritizationOutput(
                requirement_id=req_id,
                priority=priority,
                priority_score=float(data.get("priority_score", priority.score)),
                priority_rank=int(data.get("priority_rank", 1)),
                justification=coerce_str(data.get("justification", "")),
            )
        except Exception as e:
            logger.error(
                "Parse falhou | req_id=%s | erro=%s | conteúdo=%r",
                req_id,
                e,
                content[:200],
            )
            return PrioritizationOutput(
                requirement_id=req_id,
                priority=MoSCoWPriority.COULD_HAVE,
                priority_score=0.5,
                priority_rank=1,
                justification=f"Parse falhou: {e}",
                priority_parse_failed=True,
            )
