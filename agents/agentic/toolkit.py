"""Tools the agents may call, and the terminal actions with which they hand in their answer.

Each tool observes: it returns evidence and changes nothing the agent did not ask for. Whether and
when to call one is the model's decision; nothing here sequences them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agents.memory import EpisodicMemory
from domain.enums import RequirementType

RECALL_SIMILAR_CASES = "recall_similar_cases"
CHECK_CONSISTENCY = "check_consistency"
BACKLOG_OVERVIEW = "backlog_overview"
REQUEST_RECLASSIFICATION = "request_reclassification"

_NO_SIMILAR_CASES = "No similar cases handled so far."


class _RecallArgs(BaseModel):
    query: str = Field(
        description="Text to search for: similar items already handled in this batch, "
        "usually the item itself."
    )


def make_recall_tool(get_memory: Callable[[], EpisodicMemory | None]) -> StructuredTool:
    """Episodic memory read (CoALA, Sumers et al., 2023): what was decided for similar items
    earlier in this batch. The memory is looked up on each call because a new one is created
    per batch."""

    def _recall(query: str) -> str:
        memory = get_memory()
        similar = memory.retrieve_similar(query) if memory is not None else []
        if not similar:
            return _NO_SIMILAR_CASES
        return "\n".join(f'- "{i.text}" -> {i.output_summary} ({i.justification})' for i in similar)

    return StructuredTool.from_function(
        func=_recall,
        name=RECALL_SIMILAR_CASES,
        description="Recall similar items already handled in this batch and how they were decided.",
        args_schema=_RecallArgs,
    )


def structural_issues(
    requirement_type: RequirementType,
    nfr_category: str | None,
    valid_codes: Iterable[str],
) -> list[str]:
    """Contradictions between a classification and the taxonomy, checked deterministically."""
    issues: list[str] = []
    is_nfr = requirement_type is RequirementType.NON_FUNCTIONAL
    if is_nfr and nfr_category is None:
        issues.append("A non-functional requirement needs an nfr_category.")
    if not is_nfr and nfr_category is not None:
        issues.append("A functional requirement must have nfr_category null.")
    codes = sorted(set(valid_codes))
    if nfr_category is not None and nfr_category.strip().upper() not in codes:
        issues.append(f"Unknown category {nfr_category!r}. Valid codes: {', '.join(codes)}.")
    return issues


class _ConsistencyArgs(BaseModel):
    requirement_type: RequirementType
    nfr_category: str | None = None


def make_consistency_tool(valid_codes: Iterable[str]) -> StructuredTool:
    """External verification of a draft answer (CRITIC, Gou et al., 2024): the check comes from
    outside the model, so it does not depend on the model's own confidence."""
    codes = frozenset(code.upper() for code in valid_codes)

    def _check(requirement_type: RequirementType, nfr_category: str | None = None) -> str:
        issues = structural_issues(requirement_type, nfr_category, codes)
        return "Consistent." if not issues else "Inconsistent: " + " ".join(issues)

    return StructuredTool.from_function(
        func=_check,
        name=CHECK_CONSISTENCY,
        description="Check that a type and category are consistent with each other and with "
        "the taxonomy.",
        args_schema=_ConsistencyArgs,
    )


@dataclass
class BacklogView:
    """What the prioritizer can see of the batch: its composition, known from the classifier's
    reports, and the priorities decided so far. Decisions stop being recorded when the episodic
    memory freezes, so that parallel reads never race a write."""

    by_type: Counter[str]
    by_category: Counter[str]
    memory: EpisodicMemory
    decided: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.by_type.values())

    def record(self, priority: str) -> None:
        if not self.memory.frozen:
            self.decided[priority] += 1

    def render(self) -> str:
        types = ", ".join(f"{k}: {v}" for k, v in sorted(self.by_type.items()))
        categories = ", ".join(f"{k}: {v}" for k, v in sorted(self.by_category.items())) or "-"
        decided = ", ".join(f"{k}: {v}" for k, v in sorted(self.decided.items())) or "none yet"
        snapshot = " (snapshot taken after the warm-up phase)" if self.memory.frozen else ""
        return (
            f"Batch of {self.total} requirements. By type: {types}. "
            f"By NFR category: {categories}. "
            f"Priorities decided so far{snapshot}: {decided}."
        )


class _NoArgs(BaseModel):
    pass


def make_backlog_overview_tool(get_view: Callable[[], BacklogView | None]) -> StructuredTool:
    def _overview() -> str:
        view = get_view()
        return view.render() if view is not None else "No batch in progress."

    return StructuredTool.from_function(
        func=_overview,
        name=BACKLOG_OVERVIEW,
        description="Composition of the batch and the priorities already decided in it.",
        args_schema=_NoArgs,
    )


class _RequestArgs(BaseModel):
    reason: str = Field(description="Why the classification should be reconsidered.")


def make_reclassification_request_tool(send: Callable[[str], str]) -> StructuredTool:
    """Social action: asks the peer to reconsider its classification of the item at hand. The
    peer may agree or refuse; the answer arrives later as a message."""
    return StructuredTool.from_function(
        func=send,
        name=REQUEST_RECLASSIFICATION,
        description="Ask the classifier to reconsider its classification of this requirement.",
        args_schema=_RequestArgs,
    )


def make_terminal_tool(name: str, description: str, schema: type[BaseModel]) -> StructuredTool:
    """An action that ends the agent's work on an item. The runtime reads its validated
    arguments as the answer; the function body is never what carries it."""
    return StructuredTool.from_function(
        func=lambda **_: "Recorded.",
        name=name,
        description=description,
        args_schema=schema,
    )
