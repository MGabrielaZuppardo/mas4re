"""Unit tests for the NFR taxonomy lookup tool (ADR-011)."""

from __future__ import annotations

from agents.tools import make_nfr_taxonomy_tool
from domain.enums import Lang


def test_lookup_known_code_returns_description() -> None:
    tool = make_nfr_taxonomy_tool([("SE", "Security — data protection"), ("PE", "Performance")])

    result = tool.invoke({"category_code": "SE"})

    assert "Security" in result


def test_lookup_is_case_insensitive() -> None:
    tool = make_nfr_taxonomy_tool([("SE", "Security — data protection")])

    result = tool.invoke({"category_code": "se"})

    assert "Security" in result


def test_lookup_unknown_code_lists_valid_codes() -> None:
    tool = make_nfr_taxonomy_tool([("SE", "Security"), ("PE", "Performance")])

    result = tool.invoke({"category_code": "ZZ"})

    assert "Unknown category code" in result
    assert "PE" in result and "SE" in result


def test_lookup_falls_back_to_promise_default_when_categories_none() -> None:
    tool = make_nfr_taxonomy_tool(None, lang=Lang.EN)

    result = tool.invoke({"category_code": "SC"})

    assert "Unknown category code" not in result
