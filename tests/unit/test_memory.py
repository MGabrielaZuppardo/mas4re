"""Unit tests for EpisodicMemory (ADR-011)."""

from __future__ import annotations

from agents.memory import EpisodicMemory


def test_empty_memory_returns_no_similar_items() -> None:
    memory = EpisodicMemory()
    assert memory.retrieve_similar("The system must encrypt data.") == []


def test_retrieve_similar_ranks_by_lexical_overlap() -> None:
    memory = EpisodicMemory()
    memory.add("req-01", "Encrypt sensitive data records.", "NF/SE", "security concern")
    memory.add("req-02", "Generate monthly usage reports.", "F/-", "reporting feature")
    memory.add("req-03", "Encrypt sensitive data backups.", "NF/SE", "security concern")

    similar = memory.retrieve_similar("Sensitive data must be encrypted.")

    ids = [item.requirement_id for item in similar]
    # req-01 and req-03 share "sensitive"/"data" with the query; req-02 shares nothing.
    assert "req-02" not in ids
    assert set(ids) == {"req-01", "req-03"}


def test_retrieve_similar_respects_k() -> None:
    memory = EpisodicMemory(k=1)
    memory.add("req-01", "The system must encrypt data at rest.", "NF/SE", "j1")
    memory.add("req-02", "The system must encrypt data in transit.", "NF/SE", "j2")

    similar = memory.retrieve_similar("Data must be encrypted.")

    assert len(similar) == 1


def test_add_is_noop_after_freeze() -> None:
    memory = EpisodicMemory()
    memory.add("req-01", "The system must encrypt data at rest.", "NF/SE", "j1")
    memory.freeze()
    memory.add("req-02", "The system must encrypt data in transit.", "NF/SE", "j2")

    similar = memory.retrieve_similar("Data must be encrypted.")

    assert [item.requirement_id for item in similar] == ["req-01"]


def test_frozen_memory_read_only_is_safe_across_instances() -> None:
    """Freezing is per-instance state, not global -- a fresh EpisodicMemory
    (as classify_batch()/prioritize_batch() create per call) always starts
    unfrozen regardless of any previously frozen instance."""
    frozen = EpisodicMemory()
    frozen.freeze()
    fresh = EpisodicMemory()

    assert frozen.frozen is True
    assert fresh.frozen is False
