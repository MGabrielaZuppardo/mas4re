"""Which dataset a run uses, and in which language its text is read.

`dataset_name` selects the adapter; `data_lang` selects the text column. The prompt language of
a run is independent of both (`RunConfig.lang`), so a Spanish text can be processed with Portuguese
prompts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from config.settings import settings
from datasets.base import DatasetAdapter
from datasets.priorespa import PrioReSpaAdapter
from datasets.promise import PromiseAdapter
from domain.enums import Lang

PROMISE = "promise"
PRIORESPA = "priorespa"

DATASET_NAMES = (PROMISE, PRIORESPA)

_DEFAULT_DATA_LANG: dict[str, Lang] = {PROMISE: Lang.PT, PRIORESPA: Lang.ES}


def default_data_lang(dataset_name: str) -> Lang:
    """The language the dataset's own text is in (what a run reads unless told otherwise)."""
    _require_known(dataset_name)
    return _DEFAULT_DATA_LANG[dataset_name]


def default_path(dataset_name: str, data_lang: Lang | None = None) -> str:
    _require_known(dataset_name)
    if dataset_name == PROMISE:
        return settings.promise_dataset_path
    language = data_lang or default_data_lang(dataset_name)
    if language is Lang.PT:
        return settings.priorespa_pt_dataset_path
    return settings.priorespa_dataset_path


def build_adapter(
    dataset_name: str, path: str | Path | None = None, data_lang: Lang | None = None
) -> DatasetAdapter:
    _require_known(dataset_name)
    language = data_lang or default_data_lang(dataset_name)
    resolved = path or default_path(dataset_name, language)
    builders: dict[str, Callable[[], DatasetAdapter]] = {
        PROMISE: lambda: _promise(resolved, language),
        PRIORESPA: lambda: PrioReSpaAdapter(resolved, language),
    }
    return builders[dataset_name]()


def _promise(path: str | Path, data_lang: Lang) -> DatasetAdapter:
    if data_lang is not Lang.PT:
        raise ValueError(f"The PROMISE file used here has its text in PT, not {data_lang!r}")
    return PromiseAdapter(path=path)


def _require_known(dataset_name: str) -> None:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {dataset_name!r}. Available: {list(DATASET_NAMES)}")
