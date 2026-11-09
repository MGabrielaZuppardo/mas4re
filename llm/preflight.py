"""Quick check that a model can run the experiment, before a long run depends on it.

It builds the model and makes two tiny calls: a plain one, and one that has to call a tool. The
agents need tool calling, so a model that answers in prose is reported up front instead of
failing at item 80.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from llm.factory import build_llm

_PLAIN_PROMPT = "Reply with the single word OK."
_TOOL_PROMPT = "Call the echo tool with the text hello. Do not answer in prose."


@tool
def echo(text: str) -> str:
    """Return the text it was given."""
    return text


@dataclass(frozen=True)
class PreflightResult:
    model: str
    ok: bool
    tool_calling: bool
    latency_ms: float
    error: str | None = None

    def describe(self) -> str:
        if self.error:
            return f"{self.model}: FAILED, {self.error}"
        tools = "tool calling ok" if self.tool_calling else "NO tool calling"
        return f"{self.model}: reachable ({self.latency_ms:.0f} ms), {tools}"


def preflight(
    model: str, temperature: float = 0.0, llm: BaseChatModel | None = None
) -> PreflightResult:
    started = time.perf_counter()
    try:
        chat = llm or build_llm(model, temperature)
        chat.invoke(_PLAIN_PROMPT)
        reply = chat.bind_tools([echo]).invoke(_TOOL_PROMPT)  # type: ignore[attr-defined]
    except Exception as error:
        return PreflightResult(model, False, False, 0.0, f"{type(error).__name__}: {error}")
    latency_ms = (time.perf_counter() - started) * 1000
    return PreflightResult(model, True, bool(getattr(reply, "tool_calls", None)), latency_ms)
