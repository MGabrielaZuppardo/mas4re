from __future__ import annotations

import json 
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents.base import BaseAgent
from domain.enums import MoSCoWPriority, RequirementType
from domain.models import (
    BaselineOutput,
    PipelineState,
    PrioritizedRequirement,
    Requirement,
)
from llm.factory import build_llm
from prompts.v1.baseline import build_baseline_messages

logger = logging.getLogger(__name__)


class BaselineAgent(BaseAgent):
    """
    Agente baseline single-shot: classificando e priorizando em uma única chamada de LLM

        Serve como comparador para SQ2: mede o ganho do pipeline multi-agente
    sobre um agente único executando as mesmas tarefas em sequência.

    Features:
    - Dataset-agnóstico via nfr_categories injetável
    - Suporte a prompts PT/EN
    - Retry com backoff exponencial (até 3 tentativas)
    - Execução paralela via ThreadPoolExecutor
    - Saída estruturada validada com Pydantic
    """
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        nfr_categories: list[tuple[str, str]] | None = None,
        lang: str = "pt",
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        self._llm = build_llm(model, temperature)
        self._nfr_categories = nfr_categories
        self._lang = lang
        logger.info(
            "BaselineAgent inicializado | model=%s | lang=%s | nfr_categories=%s",
            model,
            lang,
            len(nfr_categories) if nfr_categories else "None",
        )

    def run(self, state: PipelineState) -> PipelineState:
        """Classifica e prioriza todos os requisitos do estado em uma única passagem."""
        logger.info(
            "Iniciando baseline | run_id=%s | n=%d",
            state.run_id,
            state.n_requirements,
        )
        prioritized = self._run_batch(state.raw_requirements, max_workers=3)
        state.prioritized_requirements = prioritized
        state.model_used = self.model
        logger.info(
            "Baseline concluído | run_id=%s | processados=%d",
            state.run_id,
            len(prioritized),
        )
        return state

    def _run_batch(
        self,
        requirements: list[Requirement],
        max_workers: int = 3,
    ) -> list[PrioritizedRequirement]:
        """Processa requisitos em paralelo e aplica ranking global."""
        results: dict[str, PrioritizedRequirement] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single, req): req
                for req in requirements
            }
            for future in as_completed(futures):
                req = futures[future]
                try:
                    results[req.id] = future.result()
                except Exception as e:
                    logger.error(
                        "Falha ao processar | id=%s | erro=%s", req.id, e
                    )

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
    def _process_single(self, requirement: Requirement) -> PrioritizedRequirement:
        """Classifica e prioriza um requisito com retry/backoff."""
        messages = build_baseline_messages(
            requirement_text=requirement.text,
            lang=self._lang,
            nfr_categories=self._nfr_categories,
        )
        response = self._llm.invoke(messages)
        output = self._parse_response(str(response.content), requirement.id)
        return PrioritizedRequirement.from_baseline(requirement, output)

    def _parse_response(self, content: str, req_id: str) -> BaselineOutput:
        """Parse do JSON retornado pelo LLM com fallback seguro."""
        try:
            clean = content.strip().strip("```json").strip("```").strip()
            data = json.loads(clean)

            req_type = RequirementType(data["requirement_type"])
            priority = MoSCoWPriority(data["priority"])

            return BaselineOutput(
                requirement_id=req_id,
                requirement_type=req_type,
                nfr_category=data.get("nfr_category"),
                confidence=float(data.get("confidence", 0.5)),
                classification_justification=data.get("classification_justification", ""),
                priority=priority,
                priority_score=float(data.get("priority_score", priority.score)),
                priority_rank=int(data.get("priority_rank", 1)),
                priority_justification=data.get("priority_justification", ""),
            )
        except Exception as e:
            logger.error(
                "Parse falhou | req_id=%s | erro=%s | conteúdo=%r",
                req_id, e, content[:200],
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
            )

