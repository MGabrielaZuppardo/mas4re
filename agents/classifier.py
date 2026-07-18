from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agents.base import BaseAgent, llm_retry
from domain.enums import Lang, RequirementType
from domain.nfr_normalization import NFR_ONLY_CODES, normalize_nfr_category

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter
from domain.models import (
    ClassificationOutput,
    ClassifiedRequirement,
    PipelineState,
    Requirement,
)
from evaluation.failure_detectors import DetectionContext
from llm.factory import build_llm
from llm.json_parser import extract_first_json
from prompts.v1.classification import build_classification_messages

# The uncertainty threshold the prioritization prompt itself assumes
# (see prompts/v1/prioritization.py) — reused here so detect_low_confidence
# flags the same requirements the pipeline already treats as uncertain.
_CONFIDENCE_THRESHOLD = 0.70

logger = logging.getLogger(__name__)


class ClassificationAgent(BaseAgent[Requirement, ClassifiedRequirement]):
    """Agente de classificação FR/NFR baseado em LLM."""

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
        logger.info(
            "ClassificationAgent inicializado | model=%s | lang=%s | nfr_categories=%s",
            model,
            lang.value,
            len(nfr_categories) if nfr_categories else "None",
        )

    def run(self, state: PipelineState) -> PipelineState:
        logger.info(
            "Iniciando classificação | run_id=%s | n=%d",
            state.run_id,
            state.n_requirements,
        )
        classified = self.classify_batch(state.raw_requirements)
        state.classified_requirements = classified
        state.model_used = self.model
        self._persist_failure_records(state)
        logger.info(
            "Classificação concluída | run_id=%s | classificados=%d",
            state.run_id,
            len(classified),
        )
        return state

    def classify_batch(
        self,
        requirements: list[Requirement],
        max_workers: int | None = None,
    ) -> list[ClassifiedRequirement]:
        return self._run_batch(requirements, stage="classify", max_workers=max_workers)

    def _detection_context(
        self,
        item: Requirement,
        output: ClassifiedRequirement | None,
        stage: str,
        parsed_ok: bool,
    ) -> DetectionContext | None:
        # NOTE: nfr_category here is already normalized (normalize_nfr_category
        # discards unrecognized values to None before this point), so
        # detect_category_hallucination will rarely fire for this stage — the
        # hallucinated category was already filtered upstream, not caught here.
        return DetectionContext(
            requirement_id=item.id,
            stage=stage,
            requirement_text=item.text,
            parsed_ok=parsed_ok,
            confidence=output.confidence if output is not None else None,
            predicted_category=output.nfr_category if output is not None else None,
            known_categories=set(NFR_ONLY_CODES),
            justification=output.justification if output is not None else "",
            confidence_threshold=_CONFIDENCE_THRESHOLD,
        )

    @llm_retry
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
        # Two distinct failure modes, deliberately handled differently:
        #   1. JSON decode failure (no structure at all) -> safe fallback,
        #      no retry, item stays in the batch (as before).
        #   2. Enum coercion / Pydantic constraint violation (structured but
        #      invalid, e.g. confidence=1.4) -> left to propagate so
        #      @llm_retry re-queries the LLM, and if it still fails after
        #      retries the item is dropped and DetectorChain sees
        #      parsed_ok=False -> SCHEMA_INVALID (see agents/base.py).
        try:
            data = json.loads(extract_first_json(content))
        except Exception as e:
            logger.error(
                "JSON decode falhou | req_id=%s | erro=%s | conteúdo=%r",
                req_id,
                e,
                content[:200],
            )
            return ClassificationOutput(
                requirement_id=req_id,
                requirement_type=RequirementType.FUNCTIONAL,
                confidence=0.0,
                justification=f"Parse falhou (JSON): {e}",
            )

        req_type = RequirementType(data["requirement_type"])
        return ClassificationOutput(
            requirement_id=req_id,
            requirement_type=req_type,
            nfr_category=normalize_nfr_category(data.get("nfr_category")),
            confidence=float(data.get("confidence", 0.5)),
            justification=data.get("justification", ""),
        )
