from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agents.base import BaseAgent, llm_retry, rank_by_priority
from domain.enums import Lang, MoSCoWPriority, RequirementType
from domain.nfr_normalization import NFR_ONLY_CODES

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter
from domain.models import (
    BaselineOutput,
    PipelineState,
    PrioritizedRequirement,
    Requirement,
)
from evaluation.failure_detectors import DetectionContext
from llm.factory import build_llm
from llm.json_parser import coerce_str, extract_first_json
from prompts.v1.baseline import build_baseline_messages

logger = logging.getLogger(__name__)

# Same uncertainty threshold the prioritization prompt assumes (see
# prompts/v1/prioritization.py / agents/classifier.py).
_CONFIDENCE_THRESHOLD = 0.70


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
        logger.info(
            "BaselineAgent inicializado | model=%s | lang=%s | nfr_categories=%s",
            model,
            lang.value,
            len(nfr_categories) if nfr_categories else "None",
        )

    def run(self, state: PipelineState) -> PipelineState:
        logger.info("Iniciando baseline | run_id=%s | n=%d", state.run_id, state.n_requirements)
        prioritized = self._prioritize_batch(state.raw_requirements)
        state.prioritized_requirements = prioritized
        state.model_used = self.model
        self._persist_failure_records(state)
        logger.info(
            "Baseline concluído | run_id=%s | processados=%d",
            state.run_id,
            len(prioritized),
        )
        return state

    def _prioritize_batch(
        self,
        requirements: list[Requirement],
        max_workers: int | None = None,
    ) -> list[PrioritizedRequirement]:
        ordered = self._run_batch(requirements, stage="baseline", max_workers=max_workers)
        return rank_by_priority(ordered)

    def _detection_context(
        self,
        item: Requirement,
        output: PrioritizedRequirement | None,
        stage: str,
        parsed_ok: bool,
    ) -> DetectionContext | None:
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
        # Same two-stage split as ClassificationAgent._parse_response — see
        # comment there for the rationale (JSON decode -> safe fallback,
        # schema/constraint violation -> propagate for @llm_retry + SCHEMA_INVALID).
        try:
            data = json.loads(extract_first_json(content))
        except Exception as e:
            logger.error(
                "JSON decode falhou | req_id=%s | erro=%s | conteúdo=%r",
                req_id,
                e,
                content[:200],
            )
            return BaselineOutput(
                requirement_id=req_id,
                requirement_type=RequirementType.FUNCTIONAL,
                nfr_category=None,
                confidence=0.0,
                classification_justification=f"Parse falhou (JSON): {e}",
                priority=MoSCoWPriority.COULD_HAVE,
                priority_score=0.5,
                priority_rank=1,
                priority_justification=f"Parse falhou (JSON): {e}",
            )

        req_type = RequirementType(data["requirement_type"])
        priority = MoSCoWPriority(data["priority"])
        return BaselineOutput(
            requirement_id=req_id,
            requirement_type=req_type,
            nfr_category=data.get("nfr_category"),
            confidence=float(data.get("confidence", 0.5)),
            classification_justification=coerce_str(data.get("classification_justification", "")),
            priority=priority,
            priority_score=float(data.get("priority_score", priority.score)),
            priority_rank=int(data.get("priority_rank", 1)),
            priority_justification=coerce_str(data.get("priority_justification", "")),
        )
