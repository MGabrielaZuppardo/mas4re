"""What the agents did, summarized from their episodes.

The figures answer, on real runs, the questions that separate an agent from a workflow: do the
episodes follow different action sequences (entropy above zero), do the agents decide to finish
or run out of budget, and how often is an emitted action invalid.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from domain.agent_actions import ActionKind, AgentEpisode, EpisodeOutcome
from domain.messages import Message, Performative

ACTIONS_FILENAME = "actions.jsonl"


def sequence_entropy(sequences: Sequence[tuple[str, ...]]) -> float:
    """Shannon entropy, in bits, of the distribution of action sequences. Zero means every
    episode took exactly the same steps."""
    if not sequences:
        return 0.0
    total = len(sequences)
    entropy = sum((n / total) * math.log2(total / n) for n in Counter(sequences).values())
    return round(entropy, 4)


def _summarize_agent(episodes: Sequence[AgentEpisode]) -> dict[str, Any]:
    actions = [action for episode in episodes for action in episode.actions]
    submitted = [e for e in episodes if e.outcome is EpisodeOutcome.SUBMITTED]
    exhausted = len(episodes) - len(submitted)
    tool_calls = Counter(
        action.name for action in actions if action.kind in (ActionKind.TOOL, ActionKind.SUBMIT)
    )
    invalid = sum(1 for action in actions if action.kind is ActionKind.INVALID)
    sequences = [episode.action_sequence for episode in episodes]
    return {
        "episodes": len(episodes),
        "submitted": len(submitted),
        "budget_exhausted": exhausted,
        "budget_exhaustion_rate": round(exhausted / len(episodes), 4),
        "submitted_via_text": sum(1 for e in submitted if e.submitted_via == "text"),
        "llm_calls": sum(e.llm_calls for e in episodes),
        "mean_llm_calls": round(sum(e.llm_calls for e in episodes) / len(episodes), 3),
        "actions": len(actions),
        "invalid_actions": invalid,
        "invalid_action_rate": round(invalid / len(actions), 4) if actions else 0.0,
        "tool_calls": dict(sorted(tool_calls.items())),
        "distinct_action_sequences": len(set(sequences)),
        "action_sequence_entropy_bits": sequence_entropy(sequences),
        "input_tokens": sum(e.input_tokens for e in episodes),
        "output_tokens": sum(e.output_tokens for e in episodes),
    }


def summarize_episodes(episodes: Sequence[AgentEpisode]) -> dict[str, dict[str, Any]]:
    """One block of figures per agent."""
    by_agent: dict[str, list[AgentEpisode]] = {}
    for episode in episodes:
        by_agent.setdefault(episode.agent, []).append(episode)
    return {agent: _summarize_agent(group) for agent, group in sorted(by_agent.items())}


def summarize_messages(messages: Sequence[Message]) -> dict[str, Any]:
    """What the agents said to each other: how many requests, who made them, and what became of
    them. The agree and refuse rates are the real-run evidence for criterion C4 (an agent that can
    refuse): both must be above zero for the "advice" not to be a disguised order."""
    requests = [m for m in messages if m.performative is Performative.REQUEST]
    answers = [m for m in messages if m.in_reply_to is not None]
    counts = Counter(m.performative for m in answers)
    answered = {m.in_reply_to for m in answers}
    decided = counts[Performative.AGREE] + counts[Performative.REFUSE]
    return {
        "requests": len(requests),
        "requests_by_sender": dict(sorted(Counter(m.sender for m in requests).items())),
        "agree": counts[Performative.AGREE],
        "refuse": counts[Performative.REFUSE],
        "failure": counts[Performative.FAILURE],
        "unanswered": sum(1 for m in requests if m.id not in answered),
        "agree_rate": round(counts[Performative.AGREE] / decided, 4) if decided else None,
        "refuse_rate": round(counts[Performative.REFUSE] / decided, 4) if decided else None,
    }


def write_episodes_jsonl(episodes: Sequence[AgentEpisode], path: Path) -> Path | None:
    """One episode per line. Written only when there are episodes: no file means the
    strategy has no agents that act."""
    if not episodes:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(episode.model_dump_json() + "\n")
    return path
