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
from domain.enums import Lang, MoSCoWPriority, NFRCategory, RequirementType

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter
from domain.models import (
    BaselineOutput,
    PipelineState,
    PrioritizedRequirement,
    Requirement,
)
from llm.factory import build_llm
from prompts.v1.baseline import build_baseline_messages

logger = logging.getLogger(__name__)

_VALID_CODES: frozenset[str] = frozenset(c.value for c in NFRCategory)
# NFR-only codes: excludes values that also appear in RequirementType (e.g. "F")
# so we never misinterpret a functional-type token as a NFR category.
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
    """Normaliza o valor bruto de nfr_category para um código PROMISE canônico.

    Aceita códigos diretos ("PE"), variantes em caixa baixa ("pe") e prosa
    em PT/EN ("desempenho", "performance"). Retorna None para valores não
    reconhecidos após log de warning.
    """
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
        results: dict[str, PrioritizedRequirement] = {}
        n = len(requirements)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._call_and_trace, "baseline", req): req for req in requirements
            }
            with tqdm(
                as_completed(futures),
                total=n,
                desc="  [baseline] ",
                unit="req",
                ncols=110,
                dynamic_ncols=False,
            ) as pbar:
                for future in pbar:
                    req = futures[future]
                    try:
                        result = future.result()
                        results[req.id] = result
                        nfr = result.nfr_category or "  -"
                        pbar.set_postfix(
                            type=result.requirement_type.value,
                            nfr=nfr,
                            priority=result.priority.value,
                            conf=f"{result.confidence:.2f}",
                        )
                    except Exception as e:
                        logger.error("Falha ao processar | id=%s | erro=%s", req.id, e)
                        pbar.set_postfix(status="ERRO")

        ordered = [results[r.id] for r in requirements if r.id in results]
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
        # C2-fix: EN conditions use the original English text.
        req_text = (
            requirement.text_en
            if self._lang is Lang.EN and requirement.text_en
            else requirement.text
        )
        messages = build_baseline_messages(
            requirement_text=req_text,
            lang=self._lang,
            nfr_categories=self._nfr_categories,
        )
        response = self._llm.invoke(messages)
        output = self._parse_response(str(response.content), requirement.id)
        return PrioritizedRequirement.from_baseline(requirement, output)

    def _parse_response(self, content: str, req_id: str) -> BaselineOutput:
        try:
            # Parser-robustness fix: use regex JSON extraction (same as
            # ClassificationAgent) so that preamble text ("Here is the JSON:")
            # or trailing explanations don't cause spurious parse failures.
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise ValueError("Nenhum JSON encontrado na resposta")
            data = json.loads(match.group())

            # C1-fix: some models (e.g. mistral:7b) place the NFR category code
            # directly in requirement_type (e.g. "PE", "SE") instead of "NF".
            # Mirror the same robustness fix applied to ClassificationAgent.
            raw_type = str(data.get("requirement_type", "")).strip().upper()
            if raw_type in _NFR_ONLY_CODES:
                logger.warning(
                    "Schema fix: requirement_type=%r interpretado como NF"
                    " + nfr_category | req_id=%s",
                    raw_type,
                    req_id,
                )
                req_type = RequirementType.NON_FUNCTIONAL
                inferred_nfr = _normalize_nfr_category(data.get("nfr_category") or raw_type)
            else:
                req_type = RequirementType(raw_type)
                inferred_nfr = _normalize_nfr_category(data.get("nfr_category"))

            priority = MoSCoWPriority(data["priority"])
            return BaselineOutput(
                requirement_id=req_id,
                requirement_type=req_type,
                nfr_category=inferred_nfr,
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
                req_id,
                e,
                content[:200],
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
