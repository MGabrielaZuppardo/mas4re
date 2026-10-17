from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

from agents.agentic.runtime import AgentRuntime, EpisodeTask, Submission
from agents.base import BaseAgent, Item, Output
from domain.agent_actions import AgentEpisode
from domain.models import PipelineState

if TYPE_CHECKING:
    from evaluation.trace_writer import TraceWriter

logger = logging.getLogger(__name__)


class AgentEpisodeError(RuntimeError):
    """The agent did not hand in an answer within its step budget. The item is lost, as with
    any other failed call, rather than filled with an invented label."""


class AgenticAgent(BaseAgent[Item, Output]):
    """An agent whose work on an item is an episode of the observe-decide-act loop.

    Subclasses give the runtime its tools and turn the submission into their output type.
    Episodes are kept until persisted, whether the item was answered or lost.
    """

    name: ClassVar[str]

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, trace_writer=trace_writer)
        self._episodes: list[AgentEpisode] = []

    def _run_episode(self, runtime: AgentRuntime, task: EpisodeTask) -> Submission:
        result = runtime.run(task)
        self._episodes.append(result.episode)
        if result.submission is None:
            raise AgentEpisodeError(
                f"{self.name} ran out of budget after {result.episode.llm_calls} model calls "
                f"without submitting an answer for requirement {task.requirement_id}"
            )
        return result.submission

    @abstractmethod
    def _runtimes(self) -> list[AgentRuntime]: ...

    def _persist_episodes(self, state: PipelineState) -> None:
        state.metrics.setdefault("agent_episodes", []).extend(
            episode.model_dump(mode="json") for episode in self._episodes
        )
        state.metrics.setdefault("agent_runtime", {})[self.name] = [
            runtime.description() for runtime in self._runtimes()
        ]
        self._episodes = []
