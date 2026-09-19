from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agents.base import BaseAgent, llm_retry, rank_by_priority
from agents.memory import EpisodicMemory, MemoryItem
from domain.enums import Lang, MoSCoWPriority

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter
from domain.models import (
    ClassifiedRequirement,
    PipelineState,
    PrioritizationOutput,
    PrioritizedRequirement,
)
from evaluation.failure_detectors import DetectionContext
from llm.factory import build_llm
from llm.json_parser import coerce_str, extract_first_json
from prompts.v1.prioritization import (
    build_prioritization_critique_messages,
    build_prioritization_messages,
)

_FEW_SHOT_HEADER_PT = "Exemplos já priorizados neste lote (referência de estilo/consistência):\n"
_FEW_SHOT_HEADER_EN = "Examples already prioritized in this batch (style/consistency reference):\n"

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
        prioritized = self.prioritize_batch(state.classified_requirements)
        state.prioritized_requirements = prioritized
        self._persist_failure_records(state)
        logger.info(
            "Priorização concluída | run_id=%s | priorizados=%d",
            state.run_id,
            len(prioritized),
        )
        return state

    def prioritize_batch(
        self,
        requirements: list[ClassifiedRequirement],
        max_workers: int | None = None,
    ) -> list[PrioritizedRequirement]:
        """Prioriza requisitos e aplica ranking global.

        Fresh memory per batch (ADR-011) -- scoped to this call.
        """
        self._memory = EpisodicMemory()
        ordered = self._run_batch(requirements, stage="prioritize", max_workers=max_workers)
        return rank_by_priority(ordered)

    def _detection_context(
        self,
        item: ClassifiedRequirement,
        output: PrioritizedRequirement | None,
        stage: str,
        parsed_ok: bool,
    ) -> DetectionContext | None:
        # MoSCoW priority isn't a confidence/category signal, so only
        # detect_schema and detect_grounding are meaningful for this stage.
        text = item.text_en if self._lang is Lang.EN and item.text_en else item.text
        return DetectionContext(
            requirement_id=item.id,
            stage=stage,
            requirement_text=text,
            parsed_ok=parsed_ok,
            justification=output.priority_justification if output is not None else "",
        )

    def _render_few_shot_block(self, requirement_text: str) -> str:
        """CoALA episodic-memory read (ADR-011) -- see
        ClassificationAgent._render_few_shot_block for the identical pattern."""
        if self._memory is None:
            return ""
        similar: list[MemoryItem] = self._memory.retrieve_similar(requirement_text)
        if not similar:
            return ""
        header = _FEW_SHOT_HEADER_PT if self._lang is Lang.PT else _FEW_SHOT_HEADER_EN
        lines = [header]
        lines.extend(
            f'- "{item.text}" -> {item.output_summary} ({item.justification})\n' for item in similar
        )
        return "".join(lines) + "\n"

    def _critique(
        self, requirement_text: str, output: PrioritizationOutput
    ) -> PrioritizationOutput:
        """Self-Refine round (Madaan et al., 2023) -- see
        ClassificationAgent._critique for the identical failure-handling
        philosophy: degrade to the original output rather than fail the item."""
        original = {
            "priority": output.priority.value,
            "priority_score": output.priority_score,
            "priority_rank": output.priority_rank,
            "justification": output.justification,
        }
        messages = build_prioritization_critique_messages(requirement_text, original, self._lang)
        response = self._llm.invoke(messages)
        try:
            data = json.loads(extract_first_json(str(response.content)))
        except Exception as e:
            logger.warning("Autocrítica: parse falhou, mantendo original | erro=%s", e)
            return output
        if not data.get("needs_revision") or not data.get("revised_output"):
            return output
        revised = data["revised_output"]
        try:
            return PrioritizationOutput(
                requirement_id=output.requirement_id,
                priority=MoSCoWPriority(revised["priority"]),
                priority_score=float(revised.get("priority_score", output.priority_score)),
                priority_rank=int(revised.get("priority_rank", output.priority_rank)),
                justification=coerce_str(revised.get("justification", output.justification)),
            )
        except Exception as e:
            logger.warning("Autocrítica: revised_output inválido, mantendo original | erro=%s", e)
            return output

    @llm_retry
    def _process_single(self, requirement: ClassifiedRequirement) -> PrioritizedRequirement:
        """Prioriza um requisito com retry/backoff."""
        req_text = (
            requirement.text_en
            if self._lang is Lang.EN and requirement.text_en
            else requirement.text
        )
        few_shot = self._render_few_shot_block(req_text)
        messages = build_prioritization_messages(
            requirement_text=req_text,
            requirement_type=requirement.requirement_type.value,
            nfr_category=requirement.nfr_category,
            lang=self._lang,
            few_shot_block=few_shot,
        )
        response = self._llm.invoke(messages)
        output = self._parse_response(str(response.content), requirement.id)
        output = self._critique(req_text, output)

        if self._memory is not None:
            self._memory.add(
                requirement.id,
                req_text,
                output_summary=output.priority.value,
                justification=output.justification,
            )
        return PrioritizedRequirement.from_classified(requirement, output)

    def _parse_response(self, content: str, req_id: str) -> PrioritizationOutput:
        """Parse do JSON retornado pelo LLM com fallback seguro.

        Same two-stage split as ClassificationAgent._parse_response — JSON
        decode failure keeps the safe fallback (no retry); an invalid enum
        or Pydantic constraint violation propagates so @llm_retry re-queries
        the LLM and, on exhaustion, DetectorChain sees parsed_ok=False.
        """
        try:
            data = json.loads(extract_first_json(content))
        except Exception as e:
            logger.error(
                "JSON decode falhou | req_id=%s | erro=%s | conteúdo=%r",
                req_id,
                e,
                content[:200],
            )
            return PrioritizationOutput(
                requirement_id=req_id,
                priority=MoSCoWPriority.COULD_HAVE,
                priority_score=0.5,
                priority_rank=1,
                justification=f"Parse falhou (JSON): {e}",
            )

        priority = MoSCoWPriority(data["priority"])
        return PrioritizationOutput(
            requirement_id=req_id,
            priority=priority,
            priority_score=float(data.get("priority_score", priority.score)),
            priority_rank=int(data.get("priority_rank", 1)),
            justification=coerce_str(data.get("justification", "")),
        )
