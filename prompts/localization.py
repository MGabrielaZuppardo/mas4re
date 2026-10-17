"""Picks the text of a prompt in the language of the run.

Every prompt constant is a mapping from `Lang` to its text. A language with no entry is an error,
never a silent fallback to another language: with a PT/EN conditional, a new language would have
been served the English prompt without anyone noticing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from domain.enums import Lang

T = TypeVar("T")


def localized(texts: Mapping[Lang, T], lang: Lang) -> T:
    if lang not in texts:
        available = ", ".join(sorted(language.value for language in texts))
        raise ValueError(f"No prompt text for language {lang.value!r}; available: {available}")
    return texts[lang]
