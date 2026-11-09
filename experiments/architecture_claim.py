"""Do the data of a run support calling it a multi-agent architecture?

The claim is decided by code, from frozen limits (`architecture_claim` in confirmatory.yml), so
that it cannot be made when the data do not bear it out. Accept and refuse rates are reported but
do not gate: a refusal rate of zero is a result (ADR-014).
"""

from __future__ import annotations

from typing import Any

from experiments.analyze_agency import AgencyRun

SUPPORTED = "supported"
MINIMAL = "minimal"
NOT_SUPPORTED = "not_supported"
INDISTINGUISHABLE = "protocols indistinguishable in practice: no protocol exchanged messages"
_INTERACTIVE_ARMS = ("P2", "P3")


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "passed": passed, "detail": detail}


def _architecture_checks(arm: str, run: AgencyRun, limits: dict[str, Any]) -> list[dict[str, Any]]:
    architecture = run.manifest.get("architecture")
    conformance = run.metrics.get("conformance")
    violations = None if conformance is None else conformance["violations"]
    return [
        _check(
            "architecture_recorded",
            architecture is not None and architecture.get("protocol") == arm.lower(),
            f"manifest architecture: {architecture and architecture.get('protocol')}",
        ),
        _check(
            "conformance",
            violations is not None and violations <= limits["max_conformance_violations"],
            f"{violations} violations (limit {limits['max_conformance_violations']})",
        ),
    ]


def _agent_checks(run: AgencyRun, limits: dict[str, Any]) -> list[dict[str, Any]]:
    activity = run.metrics.get("agent_activity") or {}
    checks = []
    for agent in ("classifier", "prioritizer"):
        block = activity.get(agent)
        if block is None:
            checks.append(_check(f"{agent}_acts", False, "no episodes recorded"))
            continue
        entropy = block["action_sequence_entropy_bits"]
        exhausted = block["budget_exhaustion_rate"]
        checks.append(
            _check(
                f"{agent}_acts",
                entropy > limits["min_action_entropy_bits"]
                and exhausted <= limits["max_budget_exhaustion_rate"],
                f"entropy {entropy} bits, budget exhaustion {exhausted}",
            )
        )
    return checks


def _interaction_checks(arm: str, run: AgencyRun, limits: dict[str, Any]) -> list[dict[str, Any]]:
    if arm not in _INTERACTIVE_ARMS:
        return []
    requests = (run.metrics.get("interaction") or {}).get("requests", 0)
    return [
        _check(
            "channel_used",
            requests >= limits["min_requests_p2_p3"],
            f"{requests} requests (minimum {limits['min_requests_p2_p3']})",
        )
    ]


def evaluate_arm(arm: str, run: AgencyRun, limits: dict[str, Any]) -> dict[str, Any]:
    checks = [
        *_architecture_checks(arm, run, limits),
        *_agent_checks(run, limits),
        *_interaction_checks(arm, run, limits),
    ]
    failed = [c["check"] for c in checks if not c["passed"]]
    interaction = run.metrics.get("interaction") or {}
    verdict = NOT_SUPPORTED if failed else (SUPPORTED if arm in _INTERACTIVE_ARMS else MINIMAL)
    return {
        "verdict": verdict,
        "failed": failed,
        "checks": checks,
        "reported": {
            "agree_rate": interaction.get("agree_rate"),
            "refuse_rate": interaction.get("refuse_rate"),
            "interaction_failures": (run.metrics.get("interaction_failures") or {}).get("total"),
        },
    }


def evaluate_arms(runs: dict[str, AgencyRun], limits: dict[str, Any]) -> dict[str, Any]:
    arms = {name: evaluate_arm(name, run, limits) for name, run in runs.items()}
    interactive = [arms[a] for a in _INTERACTIVE_ARMS if a in arms]
    unused = bool(interactive) and all("channel_used" in a["failed"] for a in interactive)
    supported = bool(interactive) and all(a["verdict"] != NOT_SUPPORTED for a in arms.values())
    return {
        "overall": SUPPORTED if supported else NOT_SUPPORTED,
        "note": INDISTINGUISHABLE if unused else None,
        "arms": arms,
    }
