"""Recompute a run's interaction metrics from its logs and compare with what it reported.

The conclusions about agency and coordination must come from `actions.jsonl`, `messages.jsonl` and
`decisions.jsonl`, not from numbers nobody can trace. This rebuilds the activity, interaction,
conformance and interaction-failure figures from those files alone and fails if any differs from
`results.json`.

Usage (from the repository root):
    python -m experiments.replay_metrics RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from domain.enums import CoordinationProtocol
from evaluation.agent_activity import summarize_episodes, summarize_messages
from evaluation.interaction_failures import summarize_interaction_failures
from evaluation.protocol_conformance import summarize_conformance
from experiments.analyze_agency import AgencyRun, load_run
from pipeline.protocol_spec import PROTOCOL_SPECS


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _replayed(run: AgencyRun) -> dict[str, Any]:
    figures: dict[str, Any] = {}
    if run.episodes:
        figures["agent_activity"] = summarize_episodes(run.episodes)
    if run.messages:
        figures["interaction"] = summarize_messages(run.messages)
    if protocol := run.manifest.get("protocol"):
        spec = PROTOCOL_SPECS[CoordinationProtocol(protocol)]
        figures["conformance"] = summarize_conformance(spec, run.messages)
        figures["interaction_failures"] = summarize_interaction_failures(
            run.messages, run.decisions
        )
    return figures


def replay(run_dir: str | Path) -> dict[str, Any]:
    run = load_run(run_dir)
    stored, replayed = run.metrics, _replayed(run)
    checked = {
        name: {"match": _normalized(figures) == stored.get(name)}
        for name, figures in replayed.items()
        if name in stored
    }
    missing = sorted(set(replayed) - set(stored))
    return {
        "run": run.path.name,
        "checked": checked,
        "not_stored": missing,
        "ok": all(c["match"] for c in checked.values()) and bool(checked),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args(argv)
    results = [replay(path) for path in args.run_dirs]
    for result in results:
        status = "OK" if result["ok"] else "DIVERGES"
        print(f"{status} {result['run']}: {sorted(result['checked'])}")
        if result["not_stored"]:
            print(f"  not in results.json: {result['not_stored']}")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
