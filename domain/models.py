from __future__ import annotations

import operator
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

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


class ClassifiedRequirement(Requirement):
    """Requisito após classificação pelo agente."""

    requirement_type: RequirementType
    nfr_category: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    justification: str = Field(default="")

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
        )


class PrioritizationOutput(BaseModel):
    """Output estruturado do agente priorizador."""

    requirement_id: str
    priority: MoSCoWPriority
    priority_score: float = Field(..., ge=0.0, le=1.0)
    priority_rank: int = Field(..., ge=1)
    justification: str = Field(default="")


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


class PrioritizedRequirement(ClassifiedRequirement):
    """Requisito após priorização pelo agente."""

    priority: MoSCoWPriority | None = None
    priority_score: float | None = None
    priority_rank: int | None = None
    priority_justification: str = Field(default="")  # ← renomeado

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
            priority=output.priority,
            priority_score=output.priority_score,
            priority_rank=output.priority_rank,
            priority_justification=output.priority_justification,  # ← renomeado
        )


class PipelineState(BaseModel):
    """Estado compartilhado entre os agentes no pipeline LangGraph."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_used: str = Field(default="")

    raw_requirements: list[Requirement] = Field(default_factory=list)
    classified_requirements: list[ClassifiedRequirement] = Field(default_factory=list)
    prioritized_requirements: list[PrioritizedRequirement] = Field(default_factory=list)

    errors: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    # ── Streaming-pipeline-only scratch fields (pipeline/graph_streaming.py) ──
    #
    # NOT touched by BaselineStrategy/TwoCallBaselineStrategy or the batch
    # pipeline/graph.py — those never populate these. Deliberately kept
    # separate from classified_requirements/prioritized_requirements above
    # (rather than annotating those with operator.add directly) because
    # classifier_node/prioritizer_node/cross_check_node return the *entire*
    # mutated PipelineState object at every step; a reducer on a public field
    # would re-add that already-accumulated value on every subsequent node's
    # return, doubling it each time. These fields are only ever written via
    # process_item_node's Send-dispatched branches (one small per-item append
    # each), and _aggregate_node reads them once to compute the final,
    # correctly-ordered/ranked values it writes into the public fields above
    # as a plain overwrite.
    stream_classified: Annotated[list[ClassifiedRequirement], operator.add] = Field(
        default_factory=list, exclude=True
    )
    stream_prioritized: Annotated[list[PrioritizedRequirement], operator.add] = Field(
        default_factory=list, exclude=True
    )
    failure_records: Annotated[list[FailureRecord], operator.add] = Field(
        default_factory=list, exclude=True
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
