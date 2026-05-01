from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from domain.enums import NFRCategory, RequirementType
from domain.models import Requirement

logger = logging.getLogger(__name__)

_LABEL_TO_TYPE: dict[str, RequirementType] = {
    "F": RequirementType.FUNCTIONAL,
    "NF": RequirementType.NON_FUNCTIONAL,
}

_LABEL_TO_CATEGORY: dict[str, NFRCategory] = {c.value: c for c in NFRCategory}

_DEFAULT_PATH = Path("datasets/data/promise_nfr/promise_nfr_pt.csv")


class PromiseAdapter:
    """
    Carrega e converte o dataset PROMISE NFR+ (versão PT) para modelos de domínio.

    O dataset PROMISE NFR contém ~625 requisitos anotados com 11 categorias
    (Cleland-Huang et al., 2007). A versão PT é a tradução para português
    brasileiro.

    Referência:
        Cleland-Huang, J. et al. (2007). A machine learning approach for
        tracing regulatory codes to product specific requirements.
        ICSE'07, IEEE.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else _DEFAULT_PATH
        self._df: pd.DataFrame | None = None
        logger.info("PromiseAdapter inicializado | path=%s", self.path)



    def load(self) -> list[Requirement]:
        """Carrega e valida o dataset. Retorna lista de Requirement."""
        logger.info("Carregando PROMISE NFR+ | path=%s", self.path)

        df = self._read_csv()
        df = self._validate_columns(df)
        df = self._clean(df)
        self._df = df

        requirements = [self._row_to_requirement(row) for _, row in df.iterrows()]

        logger.info(
            "PROMISE NFR+ carregado | total=%d | FR=%d | NFR=%d",
            len(requirements),
            sum(1 for r in requirements if r.metadata.get("label_type") == "F"),
            sum(1 for r in requirements if r.metadata.get("label_type") != "F"),
        )
        return requirements

    def load_sample(self, n: int, seed: int = 42) -> list[Requirement]:
        """Carrega amostra aleatória reprodutível de n requisitos."""
        all_reqs = self.load()
        df_sample = self._df.sample(n=min(n, len(all_reqs)), random_state=seed)  # type: ignore[union-attr]
        logger.info("Amostra carregada | n=%d | seed=%d", len(df_sample), seed)
        return [self._row_to_requirement(row) for _, row in df_sample.iterrows()]

    @property
    def dataframe(self) -> pd.DataFrame:
        """Retorna o DataFrame bruto após load()."""
        if self._df is None:
            raise RuntimeError("Chame load() antes de acessar o dataframe.")
        return self._df


    def _read_csv(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset PROMISE NFR+ não encontrado: {self.path}\n"
                "Verifique o caminho em config/settings.py → promise_dataset_path"
            )
        try:
            return pd.read_csv(self.path, encoding="utf-8")
        except Exception as e:
            logger.error("Falha ao ler CSV | path=%s | erro=%s", self.path, e)
            raise

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"RequirementText", "RequirementText_PT", "class"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Colunas obrigatórias ausentes: {missing}\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        original_len = len(df)
        df = df.dropna(subset=["RequirementText_PT", "class"])
        df = df[df["RequirementText_PT"].str.strip() != ""]
        removed = original_len - len(df)
        if removed:
            logger.warning("Linhas removidas (nulos/vazios): %d", removed)
        return df.reset_index(drop=True)

    def _row_to_requirement(self, row: pd.Series) -> Requirement:  # type: ignore[type-arg]
        label = str(row["class"]).strip().upper()
        is_functional = label == "F"
        return Requirement(
            text=str(row["RequirementText_PT"]).strip(),
            text_en=str(row.get("RequirementText", "")).strip() or None,
            source="PROMISE_NFR_PT",
            metadata={
                "label_type": "F" if is_functional else "NF",
                "label_category": label,
                "ground_truth_type": _LABEL_TO_TYPE.get("F" if is_functional else "NF"),
                "ground_truth_category": _LABEL_TO_CATEGORY.get(label),
                "project": str(row.get("ProjectID", "")).strip(),
            },
        )