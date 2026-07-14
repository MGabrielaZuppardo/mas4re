from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from domain.enums import MoSCoWPriority, RequirementType
from domain.failures import FailureRecord


class Requirement(BaseModel):
    """Requisito de software bruto- entrada do dataset ou do Elicitor"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = Field(..., min_length=5)
    text_en: str | None = Field(default=None)
    source: str = Field(default="unknown")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_nao_vazio(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("O texto do requisito não pode estar vazio.")
        return v.strip()


class ClassificationOutput(BaseModel):
    """Output estruturado do agente classificador."""

    requirement_id: str
    requirement_type: RequirementType
    nfr_category: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    justification: str = Field(default="")
    parse_failed: bool = Field(
        default=False,
        description="True quando o LLM retornou saída não-parseável e "
        "requirement_type foi preenchido com um valor de fallback, não "
        "uma classificação real. Consumido por compute_classification_metrics "
        "para excluir o item das métricas em vez de contá-lo como acerto/erro.",
    )


class ClassifiedRequirement(Requirement):
    """Requisito após classificação pelo agente."""

    requirement_type: RequirementType
    nfr_category: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    justification: str = Field(default="")
    parse_failed: bool = Field(default=False)

    @classmethod
    def from_requirement(
        cls,
        req: Requirement,
        output: ClassificationOutput,
    ) -> ClassifiedRequirement:
        return cls(
            **req.model_dump(),
            requirement_type=output.requirement_type,
            nfr_category=output.nfr_category,
            confidence=output.confidence,
            justification=output.justification,
            parse_failed=output.parse_failed,
        )


class PrioritizationOutput(BaseModel):
    """Output estruturado do agente priorizador."""

    requirement_id: str
    priority: MoSCoWPriority
    priority_score: float = Field(..., ge=0.0, le=1.0)
    priority_rank: int = Field(..., ge=1)
    justification: str = Field(default="")
    priority_parse_failed: bool = Field(
        default=False,
        description="True quando o LLM retornou saída não-parseável para a "
        "priorização e o resultado foi preenchido com um fallback fixo "
        "(COULD_HAVE, score=0.5), não uma priorização real.",
    )


class BaselineOutput(BaseModel):
    """Output estruturado do agente baseline."""

    requirement_id: str
    requirement_type: RequirementType
    nfr_category: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    classification_justification: str = Field(default="")
    priority: MoSCoWPriority
    priority_score: float = Field(..., ge=0.0, le=1.0)
    priority_rank: int = Field(..., ge=1)
    priority_justification: str = Field(default="")
    parse_failed: bool = Field(
        default=False,
        description="True quando a resposta única do baseline (classificação+"
        "priorização no mesmo JSON) não pôde ser parseada e ambos os campos "
        "foram preenchidos com fallback (FUNCTIONAL/COULD_HAVE).",
    )


class PrioritizedRequirement(ClassifiedRequirement):
    """Requisito após priorização pelo agente."""

    priority: MoSCoWPriority | None = None
    priority_score: float | None = None
    priority_rank: int | None = None
    priority_justification: str = Field(default="")  # ← renomeado
    priority_parse_failed: bool = Field(default=False)

    @classmethod
    def from_classified(
        cls,
        req: ClassifiedRequirement,
        output: PrioritizationOutput,
    ) -> PrioritizedRequirement:
        return cls(
            **req.model_dump(),
            priority=output.priority,
            priority_score=output.priority_score,
            priority_rank=output.priority_rank,
            priority_justification=output.justification,  # ← renomeado
            priority_parse_failed=output.priority_parse_failed,
        )

    @classmethod
    def from_baseline(
        cls,
        req: Requirement,
        output: BaselineOutput,
    ) -> PrioritizedRequirement:
        """Constrói a partir do output do BaselineAgent."""
        return cls(
            **req.model_dump(),
            requirement_type=output.requirement_type,
            nfr_category=output.nfr_category,
            confidence=output.confidence,
            justification=output.classification_justification,
            parse_failed=output.parse_failed,
            priority=output.priority,
            priority_score=output.priority_score,
            priority_rank=output.priority_rank,
            priority_justification=output.priority_justification,  # ← renomeado
            priority_parse_failed=output.parse_failed,
        )


class PipelineState(BaseModel):
    """Estado compartilhado entre os agentes no pipeline LangGraph."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = Field(default="")

    raw_requirements: list[Requirement] = Field(default_factory=list)
    classified_requirements: list[ClassifiedRequirement] = Field(default_factory=list)
    prioritized_requirements: list[PrioritizedRequirement] = Field(default_factory=list)

    errors: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    failure_records: list[FailureRecord] = Field(
        default_factory=list,
        description="FailureRecords emitted by DetectorChain (ADR-003) across all "
        "agents that ran in this pipeline, plus cross_check_node's "
        "inter-agent conflict records when the pipeline architecture is used.",
    )

    @property
    def n_requirements(self) -> int:
        return len(self.raw_requirements)

    @property
    def classification_rate(self) -> float:
        if not self.raw_requirements:
            return 0.0
        return len(self.classified_requirements) / len(self.raw_requirements)

    @property
    def prioritization_rate(self) -> float:
        if not self.classified_requirements:
            return 0.0
        return len(self.prioritized_requirements) / len(self.classified_requirements)
