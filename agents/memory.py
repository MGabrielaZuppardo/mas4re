"""Episodic memory for the agentic classifier/prioritizer (ADR-011).

Scoped to a single batch call (CoALA: episodic memory; Reflexion: verbal
reinforcement across attempts in the same run) -- a fresh EpisodicMemory is
created per classify_batch()/prioritize_batch() call, never persisted
between runs.

Populated only during BaseAgent._run_batch's sequential warm-up phase, then
frozen: add() becomes a no-op so the concurrent-read phase that follows is
race-free without needing a lock.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"[a-zà-ú0-9]+")


def _tokenize(text: str, min_len: int = 4) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall(text.lower()) if len(w) >= min_len)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


@dataclass(frozen=True)
class MemoryItem:
    requirement_id: str
    text: str
    output_summary: str
    justification: str


@dataclass
class EpisodicMemory:
    """Episodic memory of items already processed earlier in this batch."""

    k: int = 3
    _items: list[tuple[MemoryItem, frozenset[str]]] = field(default_factory=list, repr=False)
    frozen: bool = False

    def add(self, requirement_id: str, text: str, output_summary: str, justification: str) -> None:
        """Record a processed item. No-op once frozen()."""
        if self.frozen:
            return
        item = MemoryItem(requirement_id, text, output_summary, justification)
        self._items.append((item, _tokenize(text)))

    def freeze(self) -> None:
        """Stop accepting writes -- makes concurrent reads race-free."""
        self.frozen = True

    def retrieve_similar(self, text: str) -> list[MemoryItem]:
        """The k most lexically similar already-processed items (Jaccard over
        content-word tokens), most similar first. Zero-overlap items are
        excluded -- an unrelated example is worse than no example."""
        if not self._items:
            return []
        query = _tokenize(text)
        scored = [(_jaccard(query, tokens), item) for item, tokens in self._items]
        scored = [pair for pair in scored if pair[0] > 0]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[: self.k]]
