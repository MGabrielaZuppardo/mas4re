from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from agents.base import BaseAgent
from domain.enums import Lang, MoSCoWPriority

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter
from domain.models import (
    ClassifiedRequirement,
    PipelineState,
    PrioritizationOutput,
    PrioritizedRequirement,
)
from llm.factory import build_llm
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
        logger.info("PrioritizationAgent inicializado | model=%s | lang=%s", model, lang.value)

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
        n = len(requirements)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._call_and_trace, "prioritize", req): req
                for req in requirements
            }
            with tqdm(
                as_completed(futures),
                total=n,
                desc="  [prioritize]",
                unit="req",
                ncols=110,
                dynamic_ncols=False,
            ) as pbar:
                for future in pbar:
                    req = futures[future]
                    try:
                        result = future.result()
                        results[req.id] = result
                        pbar.set_postfix(
                            priority=result.priority.value,
                            score=f"{result.priority_score:.2f}",
                        )
                    except Exception as e:
                        logger.error("Falha ao priorizar | id=%s | erro=%s", req.id, e)
                        pbar.set_postfix(status="ERRO")

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
        # C2-fix: use original English text for EN conditions.
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
            # Parser-robustness fix: regex extraction, consistent with
            # ClassificationAgent, tolerates preamble/trailing text.
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise ValueError("Nenhum JSON encontrado na resposta")
            data = json.loads(match.group())

            priority = MoSCoWPriority(data["priority"])

            return PrioritizationOutput(
                requirement_id=req_id,
                priority=priority,
                priority_score=float(data.get("priority_score", priority.score)),
                priority_rank=int(data.get("priority_rank", 1)),
                justification=data.get("justification", ""),
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
