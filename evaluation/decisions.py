"""What each agent decided before the protocol's exchange and what stands after it.

The protocols ask agents to reconsider a decision. To tell whether that changed anything, the
decision of the first pass is kept next to the final one, with the benchmark's reference labels.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from domain.models import PrioritizedRequirement

DECISIONS_FILENAME = "decisions.jsonl"


def snapshot_decisions(items: Sequence[PrioritizedRequirement]) -> dict[str, dict[str, str | None]]:
    """First-pass type and priority of every requirement, keyed by id."""
    return {
        item.id: {
            "type": item.requirement_type.value,
            "priority": item.priority.value if item.priority else None,
        }
        for item in items
    }


def build_decisions(
    initial: dict[str, dict[str, str | None]], final: Sequence[PrioritizedRequirement]
) -> list[dict[str, Any]]:
    rows = []
    for item in final:
        first = initial.get(item.id, {})
        reference = item.reference_priority
        rows.append(
            {
                "id": item.id,
                "initial_type": first.get("type"),
                "final_type": item.requirement_type.value,
                "initial_priority": first.get("priority"),
                "final_priority": item.priority.value if item.priority else None,
                "reference_type": item.metadata.get("label_type"),
                "reference_priority": reference.value if reference else None,
            }
        )
    return rows


def write_decisions_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> Path | None:
    """One requirement per line. Written only when the first pass was recorded, which is the case
    for the agents that can revise a decision."""
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
