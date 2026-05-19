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
from domain.enums import MoSCoWPriority

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter
from domain.models import (
    ClassifiedRequirement,
    PipelineState,
    PrioritizationOutput,
    PrioritizedRequirement,
)
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
        trace_writer: TraceWriter | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, trace_writer=trace_writer)
        self._llm = build_llm(model, temperature)
        logger.info("PrioritizationAgent inicializado | model=%s", model)

    def run(self, state: PipelineState) -> PipelineState:
        """Prioriza todos os requisitos classificados do estado."""
        logger.info(
            "Iniciando priorização | run_id=%s | n=%d",
            state.run_id,
            len(state.classified_requirements),
        )
        prioritized = self.prioritize_batch(
            state.classified_requirements,
            max_workers=3,
        )
        state.prioritized_requirements = prioritized
        logger.info(
            "Priorização concluída | run_id=%s | priorizados=%d",
            state.run_id,
            len(prioritized),
        )
        return state

    def prioritize_batch(
        self,
        requirements: list[ClassifiedRequirement],
        max_workers: int = 3,
    ) -> list[PrioritizedRequirement]:
        """Prioriza requisitos em paralelo e aplica ranking global."""
        results: dict[str, PrioritizedRequirement] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._call_and_trace, "prioritize", req): req
                for req in requirements
            }
            for future in as_completed(futures):
                req = futures[future]
                try:
                    results[req.id] = future.result()
                except Exception as e:
                    logger.error("Falha ao priorizar | id=%s | erro=%s", req.id, e)

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
        messages = build_prioritization_messages(
            requirement_text=requirement.text,
            requirement_type=requirement.requirement_type.value,
            nfr_category=requirement.nfr_category,  # já é str | None
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
            )
