"""JSONL trace writer for SQ3 reproducibility (PR #22).

Um TraceEvent é emitido por requisito por chamada de agente, capturando
latência, resultado da análise e modos de falha detectados. O arquivo resultante
{run_id}.jsonl em experiments/traces/ é o artefato principal
para §5 (Protocolo Experimental) e a Disponibilidade de Artefatos do Zenodo.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_TRACE_DIR = Path("experiments") / "traces"


class TraceEvent(BaseModel):
    """One recorded agent call for a single requirement."""

    stage: str
    requirement_id: str
    model: str
    latency_ms: float
    parsed_ok: bool
    failure_modes: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TraceWriter:
    """Appends TraceEvents as newline-delimited JSON to {run_id}.jsonl.

    Designed for DI into BaseAgent: pass an instance at construction time;
    agents call _call_and_trace() which delegates here.

    Usage::

        with TraceWriter(output_dir, run_id) as tw:
            agent = ClassificationAgent(model=..., trace_writer=tw)
            agent.run(state)
        # JSONL file is flushed and closed after the with-block
    """

    def __init__(self, output_dir: Path = DEFAULT_TRACE_DIR, run_id: str = "default") -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._path = output_dir / f"{run_id}.jsonl"
        self._run_id = run_id
        self._file = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id

    def write(self, event: TraceEvent) -> None:
        payload = {"run_id": self._run_id, **event.model_dump()}
        self._file.write(json.dumps(payload, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
