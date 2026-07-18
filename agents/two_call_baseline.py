"""TwoCallBaselineAgent — ablation control for the typed inter-agent state.

PURPOSE
-------
This agent isolates the contribution of the typed PipelineState contract by
using the *same two task-specific prompts* as the pipeline architecture
(ClassificationAgent + PrioritizationAgent) but communicating between calls
via a plain Python dataclass instead of a validated Pydantic model.

What is deliberately *removed* compared to the pipeline:
  1. Pydantic model_validate at the edge transition (no schema enforcement
     between the classification and prioritization calls).
  2. Confidence-based routing: confidence is always passed as 1.0 to the
     prioritization call, so the uncertainty flag (<0.70 threshold) is
     never triggered regardless of the model's stated confidence.

What is kept identical to the pipeline:
  - Prompt templates (build_classification_messages / build_prioritization_messages)
  - Same parser logic (regex JSON extraction + _normalize_nfr_category)
  - Same concurrency (ThreadPoolExecutor, max_workers=3)
  - Same retry policy (tenacity, 3 attempts, capped exponential backoff via
    settings.retry_wait_* — see agents/base.py::llm_retry)
  - Same LLM and temperature

EXPERIMENTAL ROLE
-----------------
Comparison grid (this condition's 6 runs are part of the full 18-condition
grid: 3 models x 2 languages x {baseline, two_call_baseline, pipeline}):
  TwoCallBaseline x 3 models x 2 languages = 6 conditions

If the pipeline's F1 advantage disappears in this comparison, it is
attributable to the typed state contract (Pydantic validation + confidence
routing), not to prompt specialisation alone.  If the advantage persists,
it is attributable to the specialised prompts themselves.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tqdm import tqdm

from agents.base import BaseAgent, llm_retry, rank_by_priority
from config.settings import settings
from domain.enums import Lang, MoSCoWPriority, RequirementType
from domain.models import (
    BaselineOutput,
    PipelineState,
    PrioritizedRequirement,
    Requirement,
)
from domain.nfr_normalization import NFR_ONLY_CODES, normalize_nfr_category
from evaluation.failure_detectors import DetectionContext, default_chain
from llm.factory import build_llm
from prompts.v1.classification import build_classification_messages
from prompts.v1.prioritization import build_prioritization_messages

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter

logger = logging.getLogger(__name__)

# Same uncertainty threshold the prioritization prompt assumes (see
# prompts/v1/prioritization.py / agents/classifier.py).
_CONFIDENCE_THRESHOLD = 0.70

# Own chain instance: this agent doesn't route through BaseAgent._run_batch
# (keeps its tqdm progress loop), so it can't reuse the module-private
# instance in agents/base.py.
_DETECTOR_CHAIN = default_chain()


# ── Intermediate results: plain dataclasses, NOT Pydantic models ──────────────


@dataclass
class _ClassificationResult:
    """Unvalidated intermediate result of Call 1.

    This is the key structural difference from the pipeline: no Pydantic
    model_validate, no edge-level schema enforcement, no typed state contract.
    confidence is stored but intentionally NOT used for routing in Call 2.
    """

    requirement_type: RequirementType
    nfr_category: str | None
    confidence: float
    justification: str


@dataclass
class _PrioritizationResult:
    """Unvalidated intermediate result of Call 2."""

    priority: MoSCoWPriority
    priority_score: float
    priority_rank: int
    justification: str


# ── Main agent ────────────────────────────────────────────────────────────────


class TwoCallBaselineAgent(BaseAgent[Requirement, PrioritizedRequirement]):
    """Ablation agent: two sequential LLM calls with specialised prompts,
    no typed inter-agent state, no confidence-based routing.

    See module docstring for the full experimental rationale.
    """

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
            "TwoCallBaselineAgent inicializado | model=%s | lang=%s",
            model,
            lang.value,
        )

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def run(self, state: PipelineState) -> PipelineState:
        logger.info(
            "Iniciando two-call-baseline | run_id=%s | n=%d",
            state.run_id,
            state.n_requirements,
        )
        prioritized = self._run_batch_with_progress(
            state.raw_requirements, max_workers=settings.max_workers
        )
        state.prioritized_requirements = prioritized
        state.model_used = self.model
        self._persist_failure_records(state)
        logger.info(
            "Two-call-baseline concluído | run_id=%s | processados=%d",
            state.run_id,
            len(prioritized),
        )
        return state

    @llm_retry
    def _process_single(self, requirement: Requirement) -> PrioritizedRequirement:
        """Two sequential LLM calls with no typed state between them."""
        req_text = (
            requirement.text_en
            if self._lang is Lang.EN and requirement.text_en
            else requirement.text
        )

        # ── Call 1: Classification (same prompt as ClassificationAgent) ──────
        cls_messages = build_classification_messages(
            requirement_text=req_text,
            lang=self._lang,
            nfr_categories=self._nfr_categories,
        )
        cls_response = self._llm.invoke(cls_messages)
        cls_result = self._parse_classification(str(cls_response.content), requirement.id)

        # ── Call 2: Prioritization (same prompt as PrioritizationAgent) ──────
        # confidence is intentionally fixed at 1.0 to disable the uncertainty
        # routing mechanism (confidence < 0.70 → conservative MoSCoW label).
        # Routing is a feature of the typed state contract; removing it keeps
        # the ablation clean.  The model still receives type and nfr_category.
        pri_messages = build_prioritization_messages(
            requirement_text=req_text,
            requirement_type=cls_result.requirement_type.value,
            nfr_category=cls_result.nfr_category,
            confidence=1.0,  # routing disabled — ablation design
            lang=self._lang,
        )
        pri_response = self._llm.invoke(pri_messages)
        pri_result = self._parse_prioritization(str(pri_response.content), requirement.id)

        # ── Assemble into the shared output model ─────────────────────────────
        output = BaselineOutput(
            requirement_id=requirement.id,
            requirement_type=cls_result.requirement_type,
            nfr_category=cls_result.nfr_category,
            confidence=cls_result.confidence,  # original confidence preserved
            classification_justification=cls_result.justification,
            priority=pri_result.priority,
            priority_score=pri_result.priority_score,
            priority_rank=pri_result.priority_rank,
            priority_justification=pri_result.justification,
        )
        return PrioritizedRequirement.from_baseline(requirement, output)

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

    # ── Batch execution ───────────────────────────────────────────────────────

    def _run_batch_with_progress(
        self,
        requirements: list[Requirement],
        max_workers: int = 3,
    ) -> list[PrioritizedRequirement]:
        results: dict[str, PrioritizedRequirement] = {}
        self._failure_records = []
        n = len(requirements)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._call_and_trace, "two_call_baseline", req): req
                for req in requirements
            }
            with tqdm(
                as_completed(futures),
                total=n,
                desc="  [2call-bsl]",
                unit="req",
                ncols=110,
                dynamic_ncols=False,
            ) as pbar:
                for future in pbar:
                    req = futures[future]
                    result: PrioritizedRequirement | None = None
                    parsed_ok = True
                    try:
                        result = future.result()
                        results[req.id] = result
                        nfr = result.nfr_category or "  -"
                        priority = result.priority.value if result.priority is not None else "?"
                        pbar.set_postfix(
                            type=result.requirement_type.value,
                            nfr=nfr,
                            priority=priority,
                            conf=f"{result.confidence:.2f}",
                        )
                    except Exception as e:
                        parsed_ok = False
                        logger.error("Falha ao processar | id=%s | erro=%s", req.id, e)
                        pbar.set_postfix(status="ERRO")
                    ctx = self._detection_context(req, result, "two_call_baseline", parsed_ok)
                    if ctx is not None:
                        self._failure_records.extend(_DETECTOR_CHAIN.run(ctx))

        ordered = [results[r.id] for r in requirements if r.id in results]
        return rank_by_priority(ordered)

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_classification(self, content: str, req_id: str) -> _ClassificationResult:
        """Parse Call 1 output into an unvalidated dataclass (no Pydantic)."""
        try:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise ValueError("Nenhum JSON encontrado na resposta")
            data = json.loads(match.group())

            raw_type = str(data.get("requirement_type", "")).strip().upper()
            if raw_type in NFR_ONLY_CODES:
                logger.warning(
                    "Schema fix: requirement_type=%r interpretado como NF | req_id=%s",
                    raw_type,
                    req_id,
                )
                req_type = RequirementType.NON_FUNCTIONAL
                inferred_nfr = normalize_nfr_category(data.get("nfr_category") or raw_type)
            else:
                req_type = RequirementType(raw_type)
                inferred_nfr = normalize_nfr_category(data.get("nfr_category"))

            return _ClassificationResult(
                requirement_type=req_type,
                nfr_category=inferred_nfr,
                confidence=float(data.get("confidence", 0.5)),
                justification=data.get("justification", ""),
            )
        except Exception as e:
            logger.error(
                "Parse classificação falhou | req_id=%s | erro=%s | conteúdo=%r",
                req_id,
                e,
                content[:200],
            )
            return _ClassificationResult(
                requirement_type=RequirementType.FUNCTIONAL,
                nfr_category=None,
                confidence=0.0,
                justification=f"Parse falhou: {e}",
            )

    def _parse_prioritization(self, content: str, req_id: str) -> _PrioritizationResult:
        """Parse Call 2 output into an unvalidated dataclass (no Pydantic)."""
        try:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise ValueError("Nenhum JSON encontrado na resposta")
            data = json.loads(match.group())
            priority = MoSCoWPriority(data["priority"])
            return _PrioritizationResult(
                priority=priority,
                priority_score=float(data.get("priority_score", priority.score)),
                priority_rank=int(data.get("priority_rank", 1)),
                justification=data.get("justification", ""),
            )
        except Exception as e:
            logger.error(
                "Parse priorização falhou | req_id=%s | erro=%s | conteúdo=%r",
                req_id,
                e,
                content[:200],
            )
            return _PrioritizationResult(
                priority=MoSCoWPriority.COULD_HAVE,
                priority_score=0.5,
                priority_rank=1,
                justification=f"Parse falhou: {e}",
            )
