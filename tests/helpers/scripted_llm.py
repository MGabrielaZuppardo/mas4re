"""A chat model that replies from a script, for testing agents without a real LLM.

Each script entry is an `AIMessage`, or a function of the messages the model was shown (so a test
can make the reply depend on what the agent observed). Every call is recorded in `seen`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

ScriptEntry = AIMessage | Callable[[list[BaseMessage]], AIMessage]


def tool_call(name: str, call_id: str = "c1", **arguments: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id}],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )


def tool_calls(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"c{i}"} for i, (name, args) in enumerate(calls)
        ],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )


def text_reply(content: str) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )


class ScriptedChatModel(BaseChatModel):
    script: list[Any]
    _next: int = PrivateAttr(default=0)
    _seen: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def seen(self) -> list[list[BaseMessage]]:
        return self._seen

    @property
    def calls(self) -> int:
        return self._next

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._next >= len(self.script):
            raise AssertionError("The model was called more times than the script has replies")
        entry = self.script[self._next]
        self._next += 1
        self._seen.append(list(messages))
        reply = entry(list(messages)) if callable(entry) else entry
        return ChatResult(generations=[ChatGeneration(message=reply)])
