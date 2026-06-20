"""Unit tests for TraceWriter (PR #22).

All tests run without Ollama — agents use a stub _process_single so the
ThreadPoolExecutor path is exercised with zero LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.base import BaseAgent
from domain.enums import RequirementType
from domain.models import (
    ClassificationOutput,
    ClassifiedRequirement,
    PipelineState,
    Requirement,
)
from evaluation.trace_writer import TraceEvent, TraceWriter

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_req(text: str = "The system shall respond in under 1 second.") -> Requirement:
    return Requirement(text=text, source="test")


class _StubAgent(BaseAgent[Requirement, ClassifiedRequirement]):
    """Minimal concrete agent — no LLM, always succeeds."""

    def __init__(self, trace_writer: TraceWriter | None = None) -> None:
        super().__init__(model="stub", temperature=0.0, trace_writer=trace_writer)

    def run(self, state: PipelineState) -> PipelineState:
        return state

    def _process_single(self, item: Requirement) -> ClassifiedRequirement:
        output = ClassificationOutput(
            requirement_id=item.id,
            requirement_type=RequirementType.NON_FUNCTIONAL,
            confidence=0.9,
            justification="test",
        )
        return ClassifiedRequirement.from_requirement(item, output)


class _FailingAgent(BaseAgent[Requirement, ClassifiedRequirement]):
    """Agent that always raises to test parsed_ok=False path."""

    def __init__(self, trace_writer: TraceWriter | None = None) -> None:
        super().__init__(model="stub-fail", temperature=0.0, trace_writer=trace_writer)

    def run(self, state: PipelineState) -> PipelineState:
        return state

    def _process_single(self, item: Requirement) -> ClassifiedRequirement:
        raise ValueError("simulated LLM parse failure")


# ---------------------------------------------------------------------------
# TraceWriter — unit tests
# ---------------------------------------------------------------------------


class TestTraceWriter:
    def test_creates_file_on_init(self, tmp_path: Path) -> None:
        tw = TraceWriter(output_dir=tmp_path, run_id="r1")
        tw.close()
        assert (tmp_path / "r1.jsonl").exists()

    def test_path_property(self, tmp_path: Path) -> None:
        tw = TraceWriter(output_dir=tmp_path, run_id="abc")
        tw.close()
        assert tw.path == tmp_path / "abc.jsonl"

    def test_run_id_property(self, tmp_path: Path) -> None:
        tw = TraceWriter(output_dir=tmp_path, run_id="my-run")
        tw.close()
        assert tw.run_id == "my-run"

    def test_write_single_event(self, tmp_path: Path) -> None:
        with TraceWriter(output_dir=tmp_path, run_id="r2") as tw:
            tw.write(
                TraceEvent(
                    stage="classify",
                    requirement_id="req-1",
                    model="qwen2.5:7b",
                    latency_ms=123.45,
                    parsed_ok=True,
                )
            )
        lines = (tmp_path / "r2.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["run_id"] == "r2"
        assert payload["stage"] == "classify"
        assert payload["requirement_id"] == "req-1"
        assert payload["parsed_ok"] is True
        assert payload["latency_ms"] == pytest.approx(123.45)

    def test_write_multiple_events_appends(self, tmp_path: Path) -> None:
        with TraceWriter(output_dir=tmp_path, run_id="r3") as tw:
            for i in range(5):
                tw.write(
                    TraceEvent(
                        stage="prioritize",
                        requirement_id=f"req-{i}",
                        model="llama3.1:8b",
                        latency_ms=float(i * 10),
                        parsed_ok=True,
                    )
                )
        lines = (tmp_path / "r3.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        for line in lines:
            assert json.loads(line)["run_id"] == "r3"

    def test_write_failure_modes(self, tmp_path: Path) -> None:
        with TraceWriter(output_dir=tmp_path, run_id="r4") as tw:
            tw.write(
                TraceEvent(
                    stage="classify",
                    requirement_id="req-x",
                    model="qwen2.5:7b",
                    latency_ms=50.0,
                    parsed_ok=False,
                    failure_modes=["schema_invalid"],
                )
            )
        payload = json.loads((tmp_path / "r4.jsonl").read_text(encoding="utf-8"))
        assert payload["parsed_ok"] is False
        assert payload["failure_modes"] == ["schema_invalid"]

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        with TraceWriter(output_dir=nested, run_id="r5") as tw:
            tw.write(
                TraceEvent(
                    stage="baseline",
                    requirement_id="req-1",
                    model="phi3.5:3.8b",
                    latency_ms=10.0,
                    parsed_ok=True,
                )
            )
        assert (nested / "r5.jsonl").exists()

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        with TraceWriter(output_dir=tmp_path, run_id="r6") as tw:
            for i in range(3):
                tw.write(
                    TraceEvent(
                        stage="classify",
                        requirement_id=f"r{i}",
                        model="stub",
                        latency_ms=1.0,
                        parsed_ok=True,
                    )
                )
        for line in (tmp_path / "r6.jsonl").read_text().splitlines():
            json.loads(line)  # raises if invalid

    def test_context_manager_closes_file(self, tmp_path: Path) -> None:
        tw = TraceWriter(output_dir=tmp_path, run_id="r7")
        with tw:
            pass
        assert tw._file.closed

    def test_close_idempotent(self, tmp_path: Path) -> None:
        tw = TraceWriter(output_dir=tmp_path, run_id="r8")
        tw.close()
        tw.close()  # must not raise


# ---------------------------------------------------------------------------
# BaseAgent._call_and_trace — unit tests
# ---------------------------------------------------------------------------


class TestCallAndTrace:
    def test_no_trace_writer_still_returns_result(self) -> None:
        agent = _StubAgent(trace_writer=None)
        req = _make_req()
        result = agent._call_and_trace("classify", req)
        assert isinstance(result, ClassifiedRequirement)

    def test_emits_event_on_success(self, tmp_path: Path) -> None:
        with TraceWriter(output_dir=tmp_path, run_id="base-ok") as tw:
            agent = _StubAgent(trace_writer=tw)
            req = _make_req()
            agent._call_and_trace("classify", req)

        payload = json.loads((tmp_path / "base-ok.jsonl").read_text())
        assert payload["stage"] == "classify"
        assert payload["requirement_id"] == req.id
        assert payload["model"] == "stub"
        assert payload["parsed_ok"] is True
        assert payload["latency_ms"] > 0

    def test_emits_parsed_ok_false_on_exception(self, tmp_path: Path) -> None:
        with TraceWriter(output_dir=tmp_path, run_id="base-fail") as tw:
            agent = _FailingAgent(trace_writer=tw)
            req = _make_req()
            with pytest.raises(ValueError):
                agent._call_and_trace("classify", req)

        payload = json.loads((tmp_path / "base-fail.jsonl").read_text())
        assert payload["parsed_ok"] is False

    def test_latency_is_positive(self, tmp_path: Path) -> None:
        with TraceWriter(output_dir=tmp_path, run_id="lat") as tw:
            agent = _StubAgent(trace_writer=tw)
            agent._call_and_trace("classify", _make_req())

        payload = json.loads((tmp_path / "lat.jsonl").read_text())
        assert payload["latency_ms"] > 0

    def test_batch_emits_one_event_per_requirement(self, tmp_path: Path) -> None:
        """classify_batch() must produce one JSONL line per requirement."""
        from agents.classifier import ClassificationAgent

        reqs = [_make_req(f"req text {i}") for i in range(4)]

        with patch("agents.classifier.build_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content='{"requirement_type":"F","confidence":0.9,"justification":"ok"}'
            )
            mock_build.return_value = mock_llm

            with TraceWriter(output_dir=tmp_path, run_id="batch") as tw:
                agent = ClassificationAgent(model="stub", trace_writer=tw)
                agent.classify_batch(reqs, max_workers=2)

        lines = (tmp_path / "batch.jsonl").read_text().splitlines()
        assert len(lines) == 4
        stages = {json.loads(raw)["stage"] for raw in lines}
        assert stages == {"classify"}
