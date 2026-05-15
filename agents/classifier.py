from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents.base import BaseAgent
from domain.enums import Lang, RequirementType
from domain.models import (
    ClassificationOutput,
    ClassifiedRequirement,
    PipelineState,
    Requirement,
)
from llm.factory import build_llm
from prompts.v1.classification import build_classification_messages

logger = logging.getLogger(__name__)


class ClassificationAgent(BaseAgent[Requirement, ClassifiedRequirement]):
    """Agente de classificação FR/NFR baseado em LLM."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        nfr_categories: list[tuple[str, str]] | None = None,
        lang: Lang = Lang.PT,
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        self._llm = build_llm(model, temperature)
        self._nfr_categories = nfr_categories
        self._lang: Lang = lang
        logger.info(
            "ClassificationAgent inicializado | model=%s | lang=%s | nfr_categories=%s",
            model, lang.value,
            len(nfr_categories) if nfr_categories else "None",
        )

    def run(self, state: PipelineState) -> PipelineState:
        logger.info(
            "Iniciando classificação | run_id=%s | n=%d",
            state.run_id,
            state.n_requirements,
        )
        classified = self.classify_batch(state.raw_requirements, max_workers=3)
        state.classified_requirements = classified
        state.model_used = self.model
        logger.info(
            "Classificação concluída | run_id=%s | classificados=%d",
            state.run_id,
            len(classified),
        )
        return state

    def classify_batch(
        self,
        requirements: list[Requirement],
        max_workers: int = 3,
    ) -> list[ClassifiedRequirement]:
        results: dict[str, ClassifiedRequirement] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_single, req): req for req in requirements}
            for future in as_completed(futures):
                req = futures[future]
                try:
                    results[req.id] = future.result()
                except Exception as e:
                    logger.error("Falha ao classificar | id=%s | erro=%s", req.id, e)
        return [results[r.id] for r in requirements if r.id in results]

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=30, min=30, max=240),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _process_single(self, requirement: Requirement) -> ClassifiedRequirement:
        messages = build_classification_messages(
            requirement_text=requirement.text,
            lang=self._lang,
            nfr_categories=self._nfr_categories,
        )
        response = self._llm.invoke(messages)
        output = self._parse_response(str(response.content), requirement.id)
        return ClassifiedRequirement.from_requirement(requirement, output)

    def _parse_response(self, content: str, req_id: str) -> ClassificationOutput:
        try:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise ValueError("Nenhum JSON encontrado na resposta")
            data = json.loads(match.group())
            req_type = RequirementType(data["requirement_type"])
            return ClassificationOutput(
                requirement_id=req_id,
                requirement_type=req_type,
                nfr_category=data.get("nfr_category"),
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