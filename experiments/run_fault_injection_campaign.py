"""Systematic fault-injection campaign for the validation boundary (paper §6.6).

Replaces the five-value probe with a catalogue of faults injected into the
model's JSON output, run through the real agents of each architecture with a
scripted LLM (no model server, no cost). For every (architecture, fault) it
records where the fault is detected, whether the requirement survives, how many
LLM calls are consumed (retries included) and whether an out-of-range value
reaches the final output.

Usage (from the repository root):
    python experiments/run_fault_injection_campaign.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

from tenacity import stop_after_attempt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.baseline import BaselineAgent
from agents.classifier import ClassificationAgent
from agents.prioritizer import PrioritizationAgent
from agents.two_call_baseline import TwoCallBaselineAgent
from domain.models import PrioritizedRequirement, Requirement

OUTPUT_PATH = Path(__file__).resolve().parent / "results" / "fault_injection_campaign.json"
MODEL = "ollama/qwen2.5:7b"
MAX_ATTEMPTS = 3

CLS_OK = {
    "requirement_type": "NF",
    "nfr_category": "SE",
    "confidence": 0.9,
    "justification": "ok",
}
PRI_OK = {"priority": "M", "priority_score": 0.9, "priority_rank": 1, "justification": "ok"}


@dataclass(frozen=True)
class Fault:
    name: str
    stage: str  # "classification" or "prioritization"
    description: str
    cls_override: dict | str | None = None
    pri_override: dict | str | None = None
    value_range: bool = False


FAULTS: tuple[Fault, ...] = (
    Fault(
        "confidence_above_1",
        "classification",
        "confidence = 1.4",
        cls_override={"confidence": 1.4},
        value_range=True,
    ),
    Fault(
        "confidence_negative",
        "classification",
        "confidence = -0.5",
        cls_override={"confidence": -0.5},
        value_range=True,
    ),
    Fault(
        "confidence_nan",
        "classification",
        "confidence = NaN",
        cls_override={"confidence": float("nan")},
        value_range=True,
    ),
    Fault(
        "confidence_wrong_type",
        "classification",
        'confidence = "high"',
        cls_override={"confidence": "high"},
    ),
    Fault(
        "type_invalid_enum",
        "classification",
        'requirement_type = "X"',
        cls_override={"requirement_type": "X"},
    ),
    Fault(
        "type_missing",
        "classification",
        "requirement_type absent",
        cls_override={"requirement_type": None},
    ),
    Fault(
        "classification_malformed_json",
        "classification",
        "truncated JSON",
        cls_override='{"requirement_type": "NF", "confidence": 0.9',
    ),
    Fault(
        "score_above_1",
        "prioritization",
        "priority_score = 1.5",
        pri_override={"priority_score": 1.5},
        value_range=True,
    ),
    Fault(
        "score_negative",
        "prioritization",
        "priority_score = -0.1",
        pri_override={"priority_score": -0.1},
        value_range=True,
    ),
    Fault(
        "rank_zero",
        "prioritization",
        "priority_rank = 0",
        pri_override={"priority_rank": 0},
        value_range=True,
    ),
    Fault(
        "priority_invalid_enum", "prioritization", 'priority = "Z"', pri_override={"priority": "Z"}
    ),
    Fault("priority_missing", "prioritization", "priority absent", pri_override={"priority": None}),
    Fault(
        "prioritization_malformed_json",
        "prioritization",
        "truncated JSON",
        pri_override='{"priority": "M", "priority_score": 0.9',
    ),
)


def _render(base: dict, override: dict | str | None) -> str:
    if isinstance(override, str):
        return override
    payload = dict(base)
    for key, value in (override or {}).items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return json.dumps(payload)


class ScriptedLLM:
    """Returns the scripted responses cyclically and counts every invocation."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def invoke(self, messages):  # noqa: ARG002
        from unittest.mock import MagicMock

        response = MagicMock()
        response.content = self._responses[self.calls % len(self._responses)]
        self.calls += 1
        return response


class _ParseFailureProbe(logging.Handler):
    """Records the LLM-call index at which an agent first flagged a parse failure."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.first_call: int | None = None
        self.llm: ScriptedLLM | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if self.first_call is None and self.llm is not None and "falhou" in record.getMessage():
            self.first_call = self.llm.calls


@dataclass
class Outcome:
    architecture: str
    fault: str
    stage: str
    value_range: bool
    detected_at_call: int | None
    retained: bool
    flagged_fallback: bool
    llm_calls: int
    out_of_range_in_output: bool
    exception: str | None


def _out_of_range(result: PrioritizedRequirement | None) -> bool:
    if result is None:
        return False
    checks = (
        result.confidence is not None and not 0.0 <= result.confidence <= 1.0,
        result.priority_score is not None and not 0.0 <= result.priority_score <= 1.0,
        result.priority_rank is not None and result.priority_rank < 1,
    )
    return any(checks)


def _flagged(result: PrioritizedRequirement | None) -> bool:
    if result is None:
        return False
    return "falhou" in (result.justification or "") + (result.priority_justification or "")


def _build(agent_cls, llm: ScriptedLLM):
    with patch(f"{agent_cls.__module__}.build_llm"):
        agent = agent_cls(model=MODEL)
    agent._llm = llm
    return agent


def _no_backoff(*agent_classes) -> None:
    for agent_cls in agent_classes:
        agent_cls._process_single.retry.sleep = lambda _seconds: None


def _run_pipeline(requirement: Requirement, responses: list[str], probe) -> tuple:
    llm = ScriptedLLM(responses)
    probe.llm = llm
    classifier = _build(ClassificationAgent, llm)
    prioritizer = _build(PrioritizationAgent, llm)
    try:
        classified = classifier._process_single(requirement)
        result = prioritizer._process_single(classified)
        return result, llm.calls, None
    except Exception as exc:  # noqa: BLE001
        return None, llm.calls, type(exc).__name__


def _run_two_call(requirement: Requirement, responses: list[str], probe) -> tuple:
    llm = ScriptedLLM(responses)
    probe.llm = llm
    agent = _build(TwoCallBaselineAgent, llm)
    try:
        return agent._process_single(requirement), llm.calls, None
    except Exception as exc:  # noqa: BLE001
        return None, llm.calls, type(exc).__name__


def _run_baseline(requirement: Requirement, responses: list[str], probe) -> tuple:
    llm = ScriptedLLM(responses)
    probe.llm = llm
    agent = _build(BaselineAgent, llm)
    try:
        return agent._process_single(requirement), llm.calls, None
    except Exception as exc:  # noqa: BLE001
        return None, llm.calls, type(exc).__name__


_AGENTS = (ClassificationAgent, PrioritizationAgent, TwoCallBaselineAgent, BaselineAgent)


def _first_attempt_detection(runner, requirement, responses) -> int | None:
    """Call index at which the first attempt flags or raises on the fault."""
    probe = _ParseFailureProbe()
    root = logging.getLogger()
    root.addHandler(probe)
    original_stops = [agent._process_single.retry.stop for agent in _AGENTS]
    for agent in _AGENTS:
        agent._process_single.retry.stop = stop_after_attempt(1)
    try:
        result, calls, exception = runner(requirement, responses, probe)
    finally:
        root.removeHandler(probe)
        for agent, stop in zip(_AGENTS, original_stops):
            agent._process_single.retry.stop = stop
    if probe.first_call is not None:
        return probe.first_call
    if exception is not None or result is None:
        return calls
    return None


def run_campaign() -> list[Outcome]:
    requirement = Requirement(id="req-01", text="O sistema deve ser seguro.")
    architectures = {
        "baseline": (_run_baseline, lambda c, p: [_render({**CLS_OK, **PRI_OK}, _merge(c, p))]),
        "two_call": (
            _run_two_call,
            lambda c, p: [_render(CLS_OK, c), _render(PRI_OK, p)],
        ),
        "pipeline": (
            _run_pipeline,
            lambda c, p: [_render(CLS_OK, c), _render(PRI_OK, p)],
        ),
    }
    _no_backoff(*_AGENTS)
    outcomes: list[Outcome] = []
    for fault in FAULTS:
        for name, (runner, build_responses) in architectures.items():
            responses = build_responses(fault.cls_override, fault.pri_override)
            detected = _first_attempt_detection(runner, requirement, responses)
            probe = _ParseFailureProbe()
            result, calls, exception = runner(requirement, responses, probe)
            outcomes.append(
                Outcome(
                    architecture=name,
                    fault=fault.name,
                    stage=fault.stage,
                    value_range=fault.value_range,
                    detected_at_call=detected,
                    retained=result is not None,
                    flagged_fallback=_flagged(result),
                    llm_calls=calls,
                    out_of_range_in_output=_out_of_range(result),
                    exception=exception,
                )
            )
    return outcomes


def _merge(cls_override, pri_override) -> dict | str | None:
    if isinstance(cls_override, str):
        return cls_override
    if isinstance(pri_override, str):
        return pri_override
    return {**(cls_override or {}), **(pri_override or {})}


def summarise(outcomes: list[Outcome]) -> dict:
    summary: dict = {}
    for architecture in ("baseline", "two_call", "pipeline"):
        rows = [o for o in outcomes if o.architecture == architecture]
        for label, subset in (
            ("all", rows),
            ("value_range", [o for o in rows if o.value_range]),
            ("parse_level", [o for o in rows if not o.value_range]),
        ):
            summary.setdefault(architecture, {})[label] = {
                "faults": len(subset),
                "retained_flagged": sum(o.retained and o.flagged_fallback for o in subset),
                "dropped": sum(not o.retained for o in subset),
                "silently_accepted": sum(o.retained and not o.flagged_fallback for o in subset),
                "out_of_range_in_output": sum(o.out_of_range_in_output for o in subset),
                "mean_llm_calls": round(sum(o.llm_calls for o in subset) / max(len(subset), 1), 2),
                "mean_detected_at_call": round(
                    sum(o.detected_at_call or 0 for o in subset) / max(len(subset), 1), 2
                ),
            }
    return summary


def main() -> None:
    outcomes = run_campaign()
    summary = summarise(outcomes)
    payload = {"faults": [asdict(o) for o in outcomes], "summary": summary}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    header = f"{'architecture':<10} {'subset':<12} faults flagged dropped silent oor calls det@"
    print(header)
    for architecture, groups in summary.items():
        for label, stats in groups.items():
            print(
                f"{architecture:<10} {label:<12} {stats['faults']:>6} "
                f"{stats['retained_flagged']:>7} {stats['dropped']:>7} "
                f"{stats['silently_accepted']:>6} {stats['out_of_range_in_output']:>3} "
                f"{stats['mean_llm_calls']:>5} {stats['mean_detected_at_call']:>4}"
            )
    print(f"\nwritten: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
