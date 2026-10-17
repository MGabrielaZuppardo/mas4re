from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from datasets.base import DatasetAdapter
from domain.enums import Lang, MoSCoWPriority, NFRCategory, RequirementType
from domain.models import Requirement

logger = logging.getLogger(__name__)

SOURCE = "PRIORESPA"

_ORIGINAL_TEXT = "issue_description_final"
_TEXT_COLUMN: dict[Lang, str] = {Lang.ES: _ORIGINAL_TEXT, Lang.PT: "issue_description_pt"}
_REQUIRED = {
    "issue_id",
    _ORIGINAL_TEXT,
    "f_nf_classification",
    "relevance",
    "moscow_categorization",
}
_MODULE_TAG = re.compile(r"^\s*\[([^\]]+)\]")


def _parse_flag(value: object) -> bool | None:
    """The CSV mixes True/False with blanks; a blank means the dataset did not say."""
    if value is None or pd.isna(value):
        return None
    return str(value).strip().lower() == "true"


class PrioReSpaAdapter(DatasetAdapter):
    """PrioReSpa: issues of a real system, in Spanish, labelled with functional/non-functional
    type, PROMISE NFR category, MoSCoW priority and relevance.

    `moscow_categorization` runs from 1 to 4 and maps onto MoSCoW as 4=Must, 3=Should, 2=Could,
    1=Won't. `data_lang` picks the text column: ES is the original; PT is a translation kept in a
    second column (`issue_description_pt`) of a derived file, with the original alongside.
    """

    def __init__(self, path: Path | str, data_lang: Lang = Lang.ES) -> None:
        if data_lang not in _TEXT_COLUMN:
            raise ValueError(f"PrioReSpa has text in {sorted(_TEXT_COLUMN)}, not {data_lang!r}")
        self.path = Path(path)
        self.data_lang = data_lang
        self._df: pd.DataFrame | None = None
        logger.info("PrioReSpaAdapter inicializado | path=%s | data_lang=%s", self.path, data_lang)

    @property
    def name(self) -> str:
        return "PrioReSpa"

    def load(self) -> list[Requirement]:
        self._df = self._read()
        return [self._to_requirement(row) for _, row in self._df.iterrows()]

    def load_sample(self, n: int, seed: int = 42) -> list[Requirement]:
        self.load()
        assert self._df is not None
        sample = self._df.sample(n=min(n, len(self._df)), random_state=seed)
        logger.info("Amostra carregada | n=%d | seed=%d", len(sample), seed)
        return [self._to_requirement(row) for _, row in sample.iterrows()]

    def _read(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset PrioReSpa não encontrado: {self.path}")
        df = pd.read_csv(self.path, encoding="utf-8-sig", sep=";")
        text_column = _TEXT_COLUMN[self.data_lang]
        missing = (_REQUIRED | {text_column}) - set(df.columns)
        if missing:
            raise ValueError(
                f"Colunas obrigatórias ausentes em {self.path.name}: {sorted(missing)}\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )
        df = df.dropna(subset=[text_column, "f_nf_classification", "moscow_categorization"])
        df = df[df[text_column].str.strip() != ""]
        df["issue_id"] = df["issue_id"].astype(str).str.strip()
        if not df["issue_id"].is_unique:
            raise ValueError("issue_id deve ser único: ele vira o id estável do requisito")
        return df.reset_index(drop=True)

    def _to_requirement(self, row: pd.Series) -> Requirement:  # type: ignore[type-arg]
        label = str(row["f_nf_classification"]).strip().upper()
        is_functional = label == RequirementType.FUNCTIONAL.value
        original = str(row[_ORIGINAL_TEXT]).strip()
        module = _MODULE_TAG.match(original)
        metadata: dict[str, object] = {
            "label_type": "F" if is_functional else "NF",
            "label_category": label,
            "ground_truth_type": (
                RequirementType.FUNCTIONAL if is_functional else RequirementType.NON_FUNCTIONAL
            ),
            "ground_truth_category": NFRCategory(label),
            "ground_truth_priority": MoSCoWPriority.from_ordinal(int(row["moscow_categorization"])),
            "ground_truth_relevance": int(row["relevance"]),
            "mandatory": _parse_flag(row.get("mandatory")),
            "module": module.group(1).strip() if module else None,
        }
        if self.data_lang is Lang.PT:
            metadata["text_original_es"] = original
        return Requirement(
            id=str(row["issue_id"]).strip(),
            text=str(row[_TEXT_COLUMN[self.data_lang]]).strip(),
            source=SOURCE,
            metadata=metadata,
        )
