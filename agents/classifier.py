from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, ToolMessage

from agents.base import BaseAgent, llm_retry
from agents.memory import EpisodicMemory, MemoryItem
from agents.tools import make_nfr_taxonomy_tool
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
from llm.json_parser import coerce_str, extract_first_json
from prompts.v1.classification import (
    build_classification_critique_messages,
    build_classification_messages,
)

# The uncertainty threshold the prioritization prompt itself assumes
# (see prompts/v1/prioritization.py) — reused here so detect_low_confidence
# flags the same requirements the pipeline already treats as uncertain.
_CONFIDENCE_THRESHOLD = 0.70

_FEW_SHOT_HEADER_PT = "Exemplos já classificados neste lote (referência de estilo/consistência):\n"
_FEW_SHOT_HEADER_EN = "Examples already classified in this batch (style/consistency reference):\n"

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
        # ADR-011: agentic components. Tool is bound once (taxonomy is fixed
        # per agent instance); memory is (re)created per batch in
        # classify_batch(), scoped to one run (CoALA episodic memory).
        self._tool = make_nfr_taxonomy_tool(nfr_categories, lang)
        self._llm_with_tools = self._llm.bind_tools([self._tool])
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
        # Fresh memory per batch (ADR-011) -- scoped to this call, never
        # reused across separate classify_batch()/run() invocations.
        self._memory = EpisodicMemory()
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

    def _render_few_shot_block(self, requirement_text: str) -> str:
        """CoALA episodic-memory read: few-shot block from similar items
        already processed earlier in this batch (empty until the memory has
        something to retrieve, or once frozen with nothing similar)."""
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
        self, requirement_text: str, output: ClassificationOutput
    ) -> ClassificationOutput:
        """Self-Refine round (Madaan et al., 2023): ask the model to review
        its own answer and revise it if needed. Any failure here (parse
        error, invalid revised_output) degrades gracefully to the original,
        already-validated output rather than failing the whole item."""
        original = {
            "requirement_type": output.requirement_type.value,
            "nfr_category": output.nfr_category,
            "confidence": output.confidence,
            "justification": output.justification,
        }
        messages = build_classification_critique_messages(requirement_text, original, self._lang)
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
            return ClassificationOutput(
                requirement_id=output.requirement_id,
                requirement_type=RequirementType(revised["requirement_type"]),
                nfr_category=normalize_nfr_category(revised.get("nfr_category")),
                confidence=float(revised.get("confidence", output.confidence)),
                justification=coerce_str(revised.get("justification", output.justification)),
            )
        except Exception as e:
            logger.warning("Autocrítica: revised_output inválido, mantendo original | erro=%s", e)
            return output

    @llm_retry
    def _process_single(self, requirement: Requirement) -> ClassifiedRequirement:
        few_shot = self._render_few_shot_block(requirement.text)
        messages = build_classification_messages(
            requirement_text=requirement.text,
            lang=self._lang,
            nfr_categories=self._nfr_categories,
            few_shot_block=few_shot,
        )

        # ReAct (Yao et al., 2023): the model decides whether to call the
        # taxonomy tool; bounded to at most one call -- the follow-up
        # generation uses the unbound _llm, not _llm_with_tools, so it must
        # produce a final answer instead of requesting another tool call.
        response = self._llm_with_tools.invoke(messages)
        if response.tool_calls:
            call = response.tool_calls[0]
            tool_result = self._tool.invoke(call["args"])
            followup: list[dict[str, str] | BaseMessage] = [
                *messages,
                response,
                ToolMessage(content=str(tool_result), tool_call_id=call["id"]),
            ]
            response = self._llm.invoke(followup)

        output = self._parse_response(str(response.content), requirement.id)
        output = self._critique(requirement.text, output)

        if self._memory is not None:
            self._memory.add(
                requirement.id,
                requirement.text,
                output_summary=f"{output.requirement_type.value}/{output.nfr_category or '-'}",
                justification=output.justification,
            )
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
