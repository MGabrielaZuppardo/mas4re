from __future__ import annotations 
import uuid
from datetime import datetime
from typing import Annotated, Any

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from domain.enums import MoSCoWPriority, NFRCategory, RequirementType


class Requirement(BaseModel):
    """Requisito de software bruto — entrada do dataset ou do Elicitor."""

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
    nfr_category: NFRCategory | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    justification: str = Field(default="")

    @field_validator("nfr_category")
    @classmethod
    def categoria_obrigatoria_se_nfr(cls, v: NFRCategory | None, info: Any) -> NFRCategory | None:
        if info.data.get("requirement_type") == RequirementType.NON_FUNCTIONAL and v is None:
            raise ValueError("nfr_category é obrigatório quando requirement_type = NF.")
        return v


class ClassifiedRequirement(Requirement):
    """Requisito após classificação pelo agente."""

    requirement_type: RequirementType
    nfr_category: NFRCategory | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    justification: str = Field(default="")

    @classmethod
    def from_requirement(
        cls,
        req: Requirement,
        output: ClassificationOutput,
    ) -> "ClassifiedRequirement":
        return cls(
            **req.model_dump(),
            requirement_type=output.requirement_type,
            nfr_category=output.nfr_category,
            confidence=output.confidence,
            justification=output.justification,
        )


class PrioritizationOutput(BaseModel):
    """Output estruturado do agente priorizador."""

    requirement_id: str
    priority: MoSCoWPriority
    priority_score: float = Field(..., ge=0.0, le=1.0)
    priority_rank: int = Field(..., ge=1)
    justification: str = Field(default="")


class PrioritizedRequirement(ClassifiedRequirement):
    """Requisito após priorização pelo agente."""

    priority: MoSCoWPriority | None = None
    priority_score: float | None = None
    priority_rank: int | None = None
    justification_priority: str = Field(default="")

    @classmethod
    def from_classified(
        cls,
        req: ClassifiedRequirement,
        output: PrioritizationOutput,
    ) -> "PrioritizedRequirement":
        return cls(
            **req.model_dump(),
            priority=output.priority,
            priority_score=output.priority_score,
            priority_rank=output.priority_rank,
            justification_priority=output.justification,
        )



class PipelineState(BaseModel):
    """Estado compartilhado entre os agentes no pipeline LangGraph."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = Field(default="")

    raw_requirements: list[Requirement] = Field(default_factory=list)
    classified_requirements: list[ClassifiedRequirement] = Field(default_factory=list)
    prioritized_requirements: list[PrioritizedRequirement] = Field(default_factory=list)

    errors: Annotated[list[str], add_messages] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

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