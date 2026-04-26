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
from domain.enums import NFRCategory, RequirementType
from domain.models import (
    ClassificationOutput,
    ClassifiedRequirement,
    PipelineState,
    Requirement,
)
from llm.factory import build_llm
from prompts.v1.classification import build_classification_messages

logger = logging.getLogger(__name__)


class ClassificationAgent(BaseAgent):
    """
    Agente de classificação FR/NFR baseado em LLM.

    Classifica requisitos em Funcional ou Não-Funcional,
    atribuindo categoria NFR conforme taxonomia PROMISE NFR+.

    Features:
    - Retry com backoff exponencial (até 3 tentativas)
    - Execução paralela via ThreadPoolExecutor
    - Saída estruturada validada com Pydantic
    """

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        super().__init__(model=model, temperature=temperature)
        self._llm = build_llm(model, temperature)
        logger.info("ClassificationAgent inicializado | model=%s", model)

    def run(self, state: PipelineState) -> PipelineState:
        """Classifica todos os requisitos do estado."""
        logger.info(
            "Iniciando classificação | run_id=%s | n=%d",
            state.run_id,
            state.n_requirements,
        )
        classified = self.classify_batch(
            state.raw_requirements,
            max_workers=3,
        )
        state.classified_requirements = classified
        state.model_used = self.model
        logger.info(
            "Classificação concluída | run_id=%s | classificados=%d",
            state.run_id,
            len(classified),
        )
        return state

    # ── Batch ─────────────────────────────────────────────────────────────────

    def classify_batch(
        self,
        requirements: list[Requirement],
        max_workers: int = 3,
    ) -> list[ClassifiedRequirement]:
        """Classifica requisitos em paralelo."""
        results: dict[str, ClassifiedRequirement] = {}

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
                        "Falha ao classificar | id=%s | erro=%s", req.id, e
                    )

        # Preserva ordem original
        return [results[r.id] for r in requirements if r.id in results]

    # ── Individual ────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=30, min=30, max=240),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _process_single(self, requirement: Requirement) -> ClassifiedRequirement:
        """Classifica um requisito com retry/backoff."""
        messages = build_classification_messages(requirement.text)
        response = self._llm.invoke(messages)
        output = self._parse_response(str(response.content), requirement.id)
        return ClassifiedRequirement.from_requirement(requirement, output)

    def _parse_response(self, content: str, req_id: str) -> ClassificationOutput:
        """Parse do JSON retornado pelo LLM com fallback seguro."""
        try:
            clean = content.strip().strip("```json").strip("```").strip()
            data = json.loads(clean)

            req_type = RequirementType(data["requirement_type"])
            category_raw = data.get("nfr_category")
            category = NFRCategory(category_raw) if category_raw else None

            return ClassificationOutput(
                requirement_id=req_id,
                requirement_type=req_type,
                nfr_category=category,
                confidence=float(data.get("confidence", 0.5)),
                justification=data.get("justification", ""),
            )
        except Exception as e:
            logger.error(
                "Parse falhou | req_id=%s | erro=%s | conteúdo=%r",
                req_id, e, content[:200],
            )
            return ClassificationOutput(
                requirement_id=req_id,
                requirement_type=RequirementType.FUNCTIONAL,
                confidence=0.0,
                justification=f"Parse falhou: {e}",
            )