"""Human-readable exports of one run.

* backlog: one row per requirement the agents processed, with its classification and priority.
* failed:  requirements the agents could not process at all, with the recorded reason.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from domain.failures import FailureSeverity
from domain.models import (
    ClassifiedRequirement,
    PipelineState,
    PrioritizedRequirement,
    Requirement,
)

BACKLOG_FILENAME = "backlog.csv"
FAILED_FILENAME = "failed.csv"
DEFAULT_DELIMITER = ";"

_CSV_ENCODING = "utf-8-sig"
_GROUND_TRUTH_FIELDS = ("gold_type", "gold_category", "type_correct")


class BacklogStatus(StrEnum):
    OK = "ok"
    CLASSIFICATION_ONLY = "classification_only"


class BacklogRow(BaseModel):
    run_id: str
    requirement_id: str
    status: BacklogStatus
    priority_rank: int | None
    text: str
    text_en: str | None
    requirement_type: str
    nfr_category: str | None
    classification_confidence: float
    classification_justification: str
    priority: str | None
    priority_score: float | None
    priority_justification: str
    gold_type: str | None
    gold_category: str | None
    type_correct: bool | None


class FailedRow(BaseModel):
    """`failure_*` come from the fatal detection; `detections` lists every one recorded."""

    run_id: str
    requirement_id: str
    text: str
    text_en: str | None
    failure_stage: str | None
    failure_mode: str | None
    failure_severity: str | None
    failure_evidence: str
    detections: str
    gold_type: str | None
    gold_category: str | None


@dataclass(frozen=True)
class Backlog:
    rows: list[BacklogRow]
    failed: list[FailedRow]


def _ground_truth(requirement: Requirement) -> tuple[str | None, str | None]:
    return requirement.metadata.get("label_type"), requirement.metadata.get("label_category")


def _backlog_row(run_id: str, source: Requirement, result: ClassifiedRequirement) -> BacklogRow:
    gold_type, gold_category = _ground_truth(source)
    prioritized = result if isinstance(result, PrioritizedRequirement) else None
    priority = prioritized.priority if prioritized else None
    return BacklogRow(
        run_id=run_id,
        requirement_id=source.id,
        status=BacklogStatus.OK if priority is not None else BacklogStatus.CLASSIFICATION_ONLY,
        priority_rank=prioritized.priority_rank if prioritized else None,
        text=source.text,
        text_en=source.text_en,
        requirement_type=result.requirement_type.value,
        nfr_category=result.nfr_category,
        classification_confidence=result.confidence,
        classification_justification=result.justification,
        priority=priority.value if priority is not None else None,
        priority_score=prioritized.priority_score if prioritized else None,
        priority_justification=prioritized.priority_justification if prioritized else "",
        gold_type=gold_type,
        gold_category=gold_category,
        type_correct=None if gold_type is None else result.requirement_type.value == gold_type,
    )


def _describe_detection(record: dict[str, Any]) -> str:
    kind = ":".join(str(record.get(key, "")) for key in ("stage", "mode", "severity"))
    return f"{kind} - {record.get('evidence', '')}"


def _failed_row(run_id: str, source: Requirement, detections: list[dict[str, Any]]) -> FailedRow:
    gold_type, gold_category = _ground_truth(source)
    fatal = next((d for d in detections if d.get("severity") == FailureSeverity.FATAL.value), {})
    return FailedRow(
        run_id=run_id,
        requirement_id=source.id,
        text=source.text,
        text_en=source.text_en,
        failure_stage=fatal.get("stage"),
        failure_mode=fatal.get("mode"),
        failure_severity=fatal.get("severity"),
        failure_evidence=fatal.get("evidence", ""),
        detections=" | ".join(_describe_detection(d) for d in detections),
        gold_type=gold_type,
        gold_category=gold_category,
    )


def _detections_by_requirement(state: PipelineState) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in state.metrics.get("failure_detections", []):
        grouped.setdefault(record["requirement_id"], []).append(record)
    return grouped


def _by_rank(rows: list[BacklogRow]) -> list[BacklogRow]:
    """Highest priority first; requirements without a rank keep their input order at the end."""
    return sorted(rows, key=lambda row: (row.priority_rank is None, row.priority_rank or 0))


def build_backlog(state: PipelineState, run_id: str) -> Backlog:
    prioritized = {r.id: r for r in state.prioritized_requirements}
    classified = {r.id: r for r in state.classified_requirements}
    detections = _detections_by_requirement(state)

    rows: list[BacklogRow] = []
    failed: list[FailedRow] = []
    for source in state.raw_requirements:
        result = prioritized.get(source.id) or classified.get(source.id)
        if result is None:
            failed.append(_failed_row(run_id, source, detections.get(source.id, [])))
        else:
            rows.append(_backlog_row(run_id, source, result))
    return Backlog(rows=_by_rank(rows), failed=failed)


def backlog_from_results(results: dict[str, Any]) -> Backlog:
    """Rebuild the backlog of a finished run from its results.json.

    The requirements the agents could not process are not stored there, so `failed` is
    always empty.
    """
    run_id = results.get("run_id", "")
    predictions = [PrioritizedRequirement.model_validate(p) for p in results["predictions"]]
    rows = [_backlog_row(run_id, prediction, prediction) for prediction in predictions]
    return Backlog(rows=_by_rank(rows), failed=[])


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _columns(
    model: type[BaseModel], rows: Sequence[BaseModel], include_ground_truth: bool
) -> list[str]:
    columns = list(model.model_fields)
    has_ground_truth = any(getattr(row, "gold_type", None) is not None for row in rows)
    if include_ground_truth and has_ground_truth:
        return columns
    return [column for column in columns if column not in _GROUND_TRUTH_FIELDS]


def _write_csv(
    path: Path,
    model: type[BaseModel],
    rows: Sequence[BaseModel],
    delimiter: str,
    include_ground_truth: bool,
) -> Path:
    if len(delimiter) != 1:
        raise ValueError(f"CSV delimiter must be a single character, got {delimiter!r}")
    columns = _columns(model, rows, include_ground_truth)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=_CSV_ENCODING, newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(columns)
        for row in rows:
            values = row.model_dump()
            writer.writerow(_cell(values[column]) for column in columns)
    return path


def write_backlog_csv(
    rows: Sequence[BacklogRow],
    path: Path,
    delimiter: str = DEFAULT_DELIMITER,
    include_ground_truth: bool = True,
) -> Path:
    return _write_csv(path, BacklogRow, rows, delimiter, include_ground_truth)


def write_failed_csv(
    rows: Sequence[FailedRow],
    path: Path,
    delimiter: str = DEFAULT_DELIMITER,
    include_ground_truth: bool = True,
) -> Path | None:
    """Written only when at least one requirement failed: no file means no failures."""
    if not rows:
        return None
    return _write_csv(path, FailedRow, rows, delimiter, include_ground_truth)
