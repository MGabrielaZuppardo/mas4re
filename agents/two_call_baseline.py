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
  - Same retry policy (tenacity, 3 attempts, 30–240s backoff)
  - Same LLM and temperature

EXPERIMENTAL ROLE
-----------------
Comparison grid (added to the original 12-condition study):
  TwoCallBaseline × 3 models × 2 languages = 6 new conditions

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

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from agents.base import BaseAgent
from domain.enums import Lang, MoSCoWPriority, NFRCategory, RequirementType
from domain.failures import FailureRecord, FailureSeverity
from domain.models import (
    BaselineOutput,
    PipelineState,
    PrioritizedRequirement,
    Requirement,
)
from evaluation.failure_detectors import DetectionContext, default_chain, max_severity
from llm.factory import build_llm
from prompts.v1.classification import build_classification_messages
from prompts.v1.prioritization import build_prioritization_messages

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter

logger = logging.getLogger(__name__)

_KNOWN_NFR_CATEGORIES = {c.value for c in NFRCategory.nfr_only()}
_CONFIDENCE_UNCERTAIN_THRESHOLD = 0.70

# ── NFR category normalisation (mirrored from ClassificationAgent) ────────────

_VALID_CODES: frozenset[str] = frozenset(c.value for c in NFRCategory)
_NFR_ONLY_CODES: frozenset[str] = _VALID_CODES - frozenset(t.value for t in RequirementType)

_PROSE_TO_CODE: dict[str, str] = {
    # Português
    "disponibilidade": "A",
    "tolerância a falhas": "FT",
    "tolerancia a falhas": "FT",
    "tolerância": "FT",
    "aparência": "LF",
    "aparencia": "LF",
    "estética": "LF",
    "estetica": "LF",
    "look and feel": "LF",
    "manutenibilidade": "MN",
    "manutenção": "MN",
    "manutencao": "MN",
    "flexibilidade": "MN",
    "documentação": "MN",
    "suporte": "MN",
    "operacional": "O",
    "operacionalidade": "O",
    "tempo de market": "O",
    "formato de dados": "O",
    "desempenho": "PE",
    "tempo de resposta": "PE",
    "portabilidade": "PO",
    "compatibilidade": "PO",
    "globalização": "PO",
    "internacionalização": "PO",
    "escalabilidade": "SC",
    "segurança": "SE",
    "seguranca": "SE",
    "usabilidade": "US",
    "acessibilidade": "US",
    # English
    "availability": "A",
    "fault tolerance": "FT",
    "reliability": "FT",
    "reliabilidade": "FT",
    "confiabilidade": "FT",
    "maintainability": "MN",
    "flexibility": "MN",
    "documentation": "MN",
    "operational": "O",
    "time to market": "O",
    "data format": "O",
    "performance": "PE",
    "response time": "PE",
    "portability": "PO",
    "compatibility": "PO",
    "internationalization": "PO",
    "globalization": "PO",
    "scalability": "SC",
    "security": "SE",
    "usability": "US",
    "accessibility": "US",
}


def _normalize_nfr_category(raw: str | None) -> str | None:
    if raw is None:
        return None
    upper = raw.strip().upper()
    if upper in _VALID_CODES:
        return upper
    lower = raw.strip().lower()
    if lower in _PROSE_TO_CODE:
        code = _PROSE_TO_CODE[lower]
        logger.warning("nfr_category prosa mapeada | raw=%r -> %s", raw, code)
        return code
    logger.warning("nfr_category nao reconhecida, descartada | raw=%r", raw)
    return None


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
    parse_failed: bool = False


@dataclass
class _PrioritizationResult:
    """Unvalidated intermediate result of Call 2."""

    priority: MoSCoWPriority
    priority_score: float
    priority_rank: int
    justification: str
    parse_failed: bool = False


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
        self._detector_chain = default_chain()
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
        failed_ids: list[str] = []
        failure_records: list[FailureRecord] = []
        prioritized = self._run_batch(
            state.raw_requirements,
            max_workers=3,
            failed_ids=failed_ids,
            failure_records=failure_records,
        )
        state.prioritized_requirements = prioritized
        state.model_used = self.model
        state.errors.extend(f"two_call_baseline_failed:{req_id}" for req_id in failed_ids)
        state.failure_records.extend(failure_records)
        n_fatal = sum(1 for r in failure_records if r.severity is FailureSeverity.FATAL)
        logger.info(
            "Two-call-baseline concluído | run_id=%s | processados=%d | dropados=%d | "
            "fatal=%d | flagged=%d",
            state.run_id,
            len(prioritized),
            len(failed_ids),
            n_fatal,
            len(failure_records) - n_fatal,
        )
        return state

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=30, min=30, max=240),
        stop=stop_after_attempt(3),
        reraise=True,
    )
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
            parse_failed=cls_result.parse_failed or pri_result.parse_failed,
        )
        return PrioritizedRequirement.from_baseline(requirement, output)

    # ── Batch execution ───────────────────────────────────────────────────────

    def _run_batch(
        self,
        requirements: list[Requirement],
        max_workers: int = 3,
        failed_ids: list[str] | None = None,
        failure_records: list[FailureRecord] | None = None,
    ) -> list[PrioritizedRequirement]:
        results: dict[str, PrioritizedRequirement] = {}
        n = len(requirements)
        req_by_id = {r.id: r for r in requirements}
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
                    try:
                        output = future.result()
                    except Exception as e:
                        logger.error("Falha ao processar | id=%s | erro=%s", req.id, e)
                        pbar.set_postfix(status="ERRO")
                        if failed_ids is not None:
                            failed_ids.append(req.id)
                        continue

                    ctx = DetectionContext(
                        requirement_id=output.id,
                        stage="two_call_baseline",
                        requirement_text=req_by_id[output.id].text,
                        parsed_ok=not output.parse_failed,
                        confidence=output.confidence,
                        predicted_category=output.nfr_category,
                        known_categories=_KNOWN_NFR_CATEGORIES,
                        justification=output.justification,
                        confidence_threshold=_CONFIDENCE_UNCERTAIN_THRESHOLD,
                    )
                    records = self._detector_chain.run(ctx)
                    if failure_records is not None:
                        failure_records.extend(records)
                    if max_severity(records) is FailureSeverity.FATAL:
                        continue

                    results[req.id] = output
                    nfr = output.nfr_category or "  -"
                    pbar.set_postfix(
                        type=output.requirement_type.value,
                        nfr=nfr,
                        priority=output.priority.value,
                        conf=f"{output.confidence:.2f}",
                    )

        ordered = [results[r.id] for r in requirements if r.id in results]
        ordered.sort(key=lambda r: r.priority_score or 0.0, reverse=True)
        for rank, req in enumerate(ordered, start=1):
            req.priority_rank = rank
        return ordered

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_classification(self, content: str, req_id: str) -> _ClassificationResult:
        """Parse Call 1 output into an unvalidated dataclass (no Pydantic)."""
        try:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise ValueError("Nenhum JSON encontrado na resposta")
            data = json.loads(match.group())

            raw_type = str(data.get("requirement_type", "")).strip().upper()
            if raw_type in _NFR_ONLY_CODES:
                logger.warning(
                    "Schema fix: requirement_type=%r interpretado como NF | req_id=%s",
                    raw_type,
                    req_id,
                )
                req_type = RequirementType.NON_FUNCTIONAL
                inferred_nfr = _normalize_nfr_category(data.get("nfr_category") or raw_type)
            else:
                req_type = RequirementType(raw_type)
                inferred_nfr = _normalize_nfr_category(data.get("nfr_category"))

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
                parse_failed=True,
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
                parse_failed=True,
            )
