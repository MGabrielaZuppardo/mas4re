"""Grounding tool for the agentic classifier (ADR-011).

ReAct pattern (Yao et al., 2023): the model decides whether to call this
tool -- typically when unsure which NFR category applies -- instead of the
code forcing a lookup. Backed by the same nfr_categories taxonomy already
injected into the classification prompt (prompts/v1/classification.py ::
build_classification_messages / _shared.py::build_nfr_block), so there is a
single source of truth for category definitions.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from domain.enums import Lang
from prompts.v1._shared import default_nfr_categories


class _LookupNFRTaxonomyArgs(BaseModel):
    category_code: str = Field(description="NFR category code to look up, e.g. 'SE', 'PE'.")


def make_nfr_taxonomy_tool(
    nfr_categories: list[tuple[str, str]] | None,
    lang: Lang = Lang.PT,
) -> StructuredTool:
    """Build a lookup_nfr_taxonomy tool bound to one agent's taxonomy.

    A factory (not a module-level @tool) because the taxonomy varies per
    agent instance (dataset-provided nfr_categories, or the PROMISE default
    when None -- same fallback build_nfr_block already uses).
    """
    categories = nfr_categories or default_nfr_categories(lang)
    lookup = {code.upper(): desc for code, desc in categories}

    def _lookup_nfr_taxonomy(category_code: str) -> str:
        desc = lookup.get(category_code.strip().upper())
        if desc is None:
            valid = ", ".join(sorted(lookup))
            return f"Unknown category code {category_code!r}. Valid codes: {valid}."
        return f"{category_code.upper()}: {desc}"

    return StructuredTool.from_function(
        func=_lookup_nfr_taxonomy,
        name="lookup_nfr_taxonomy",
        description="Look up the canonical definition of an NFR category code before assigning it.",
        args_schema=_LookupNFRTaxonomyArgs,
    )
