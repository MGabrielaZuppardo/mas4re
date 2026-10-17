"""Translation validation for datasets translated into Brazilian Portuguese.

Pipeline (shown for PROMISE; PrioReSpa is the same with ES in place of EN):
    EN (original) -> [Ollama LLM] -> PT (existing translation)
                                       -> [Ollama LLM] -> EN' (back-translated)
                                                           -> BERTScore(EN, EN')

BERTScore F1 measures semantic preservation without any human reference.
Thresholds (empirical for short technical sentences):
    >= 0.90  -> accepted
    0.80-0.89 -> flagged for review
    < 0.80   -> rejected / re-translate

Usage:
    validator = TranslationValidator()
    report = validator.validate("datasets/data/promise_nfr/promise_nfr_pt.csv")
    report.save("datasets/data/promise_nfr/validation_report.csv")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_BACK_TRANSLATE_PROMPT = (
    "Translate the following text from {translated_name} to {source_name}. "
    "Return ONLY the translated text, nothing else.\n\nText: {text}"
)

_DEFAULT_MODEL = "ollama/qwen2.5:7b"
_THRESHOLD_ACCEPT = 0.90
_THRESHOLD_REVIEW = 0.80


@dataclass(frozen=True)
class TranslationPair:
    """Which columns of a dataset hold the original and its Portuguese translation.

    `source_code` is the ISO code of the original language (it also names the report columns and
    selects the BERTScore model). `metadata_columns` maps report keys to dataset columns carried
    into the report.
    """

    source_column: str
    translated_column: str
    source_code: str
    source_name: str
    translated_name: str = "Brazilian Portuguese"
    metadata_columns: tuple[tuple[str, str], ...] = ()
    separator: str = ","
    encoding: str = "utf-8"


PROMISE_EN_PT = TranslationPair(
    source_column="RequirementText",
    translated_column="RequirementText_PT",
    source_code="en",
    source_name="English",
    metadata_columns=(("project_id", "ProjectID"), ("class", "class")),
)

PRIORESPA_ES_PT = TranslationPair(
    source_column="issue_description_final",
    translated_column="issue_description_pt",
    source_code="es",
    source_name="Spanish",
    metadata_columns=(("issue_id", "issue_id"), ("class", "f_nf_classification")),
    separator=";",
    encoding="utf-8-sig",
)


@dataclass
class ValidationReport:
    """Results of the back-translation + BERTScore validation."""

    rows: list[dict] = field(default_factory=list)

    # ── aggregate stats ────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    @property
    def n_total(self) -> int:
        return len(self.rows)

    @property
    def n_accepted(self) -> int:
        return sum(1 for r in self.rows if r["decision"] == "accepted")

    @property
    def n_review(self) -> int:
        return sum(1 for r in self.rows if r["decision"] == "review")

    @property
    def n_rejected(self) -> int:
        return sum(1 for r in self.rows if r["decision"] == "rejected")

    @property
    def mean_f1(self) -> float:
        f1s = [r["bertscore_f1"] for r in self.rows]
        return sum(f1s) / len(f1s) if f1s else 0.0

    def summary(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_accepted": self.n_accepted,
            "n_review": self.n_review,
            "n_rejected": self.n_rejected,
            "mean_bertscore_f1": round(self.mean_f1, 4),
            "threshold_accept": _THRESHOLD_ACCEPT,
            "threshold_review": _THRESHOLD_REVIEW,
        }

    def save(self, path: str | Path) -> None:
        """Save per-requirement scores to CSV."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(out, index=False, encoding="utf-8")
        logger.info("Validation report saved to %s", out)

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\nValidation Summary — {s['n_total']} requirements")
        print(f"  Mean BERTScore F1 : {s['mean_bertscore_f1']:.4f}")
        print(
            f"  Accepted (>= {s['threshold_accept']}) : "
            f"{s['n_accepted']} ({s['n_accepted'] / s['n_total'] * 100:.1f}%)"
        )
        print(
            f"  Review   ({s['threshold_review']}-{s['threshold_accept']}) : "
            f"{s['n_review']} ({s['n_review'] / s['n_total'] * 100:.1f}%)"
        )
        print(
            f"  Rejected (< {s['threshold_review']}) : "
            f"{s['n_rejected']} ({s['n_rejected'] / s['n_total'] * 100:.1f}%)"
        )


class TranslationValidator:
    """Back-translation + BERTScore validator for a dataset translated into PT.

    Args:
        model:      Ollama model used for back-translation (PT -> original language).
        batch_size: Number of requirements per BERTScore batch.
        threshold_accept: Min F1 to mark a translation as accepted.
        threshold_review: Min F1 to mark as review (below = rejected).
        pair:       Columns and languages of the dataset (default: PROMISE EN -> PT).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        batch_size: int = 32,
        threshold_accept: float = _THRESHOLD_ACCEPT,
        threshold_review: float = _THRESHOLD_REVIEW,
        pair: TranslationPair = PROMISE_EN_PT,
    ) -> None:
        self._pair = pair
        self._model = model
        self._batch_size = batch_size
        self._threshold_accept = threshold_accept
        self._threshold_review = threshold_review
        self._llm = self._build_llm()

    def _build_llm(self):  # type: ignore[return]
        from llm.factory import build_llm

        return build_llm(self._model, temperature=0.0)

    # ── back-translation ───────────────────────────────────────────────────

    def back_translate(self, texts_pt: list[str]) -> list[str]:
        """Translate a list of PT texts back to the original language using the LLM."""
        from tqdm import tqdm

        results: list[str] = []
        for text in tqdm(
            texts_pt, desc=f"Back-translating PT→{self._pair.source_code}", unit="req"
        ):
            prompt = _BACK_TRANSLATE_PROMPT.format(
                translated_name=self._pair.translated_name,
                source_name=self._pair.source_name,
                text=text,
            )
            try:
                response = self._llm.invoke(prompt)
                results.append(str(response.content).strip())
            except Exception as e:
                logger.warning("Back-translation failed for %r: %s", text[:60], e)
                results.append("")
        return results

    # ── BERTScore ──────────────────────────────────────────────────────────

    def compute_bertscore(
        self,
        originals: list[str],
        back_translated: list[str],
    ) -> tuple[list[float], list[float], list[float]]:
        """Return (precision, recall, F1) per requirement."""
        from bert_score import score as bert_score

        logger.info("Computing BERTScore for %d pairs...", len(originals))
        P, R, F1 = bert_score(
            cands=back_translated,
            refs=originals,
            lang=self._pair.source_code,
            batch_size=self._batch_size,
            verbose=False,
        )
        return P.tolist(), R.tolist(), F1.tolist()

    # ── decision ───────────────────────────────────────────────────────────

    def _decide(self, f1: float) -> str:
        if f1 >= self._threshold_accept:
            return "accepted"
        if f1 >= self._threshold_review:
            return "review"
        return "rejected"

    # ── main entry point ───────────────────────────────────────────────────

    def validate(self, dataset_path: str | Path) -> ValidationReport:
        """Run the full validation pipeline on a translated CSV.

        Returns a ValidationReport with per-requirement scores and decisions.
        """
        pair = self._pair
        path = Path(dataset_path)
        logger.info("Loading dataset from %s", path)
        df = pd.read_csv(path, encoding=pair.encoding, sep=pair.separator)

        required = {pair.source_column, pair.translated_column}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")

        df = df.dropna(subset=[pair.source_column, pair.translated_column])
        originals = df[pair.source_column].tolist()
        translated = df[pair.translated_column].tolist()

        logger.info("Back-translating %d requirements...", len(translated))
        back_translated = self.back_translate(translated)

        P, R, F1 = self.compute_bertscore(originals, back_translated)

        report = ValidationReport()
        for i, (orig, trans, back, p, r, f1) in enumerate(
            zip(originals, translated, back_translated, P, R, F1)
        ):
            report.rows.append(
                {
                    "index": i,
                    **{
                        key: df[column].iloc[i] if column in df.columns else ""
                        for key, column in pair.metadata_columns
                    },
                    f"text_{pair.source_code}_original": orig,
                    "text_pt_translated": trans,
                    f"text_{pair.source_code}_back": back,
                    "bertscore_precision": round(p, 4),
                    "bertscore_recall": round(r, 4),
                    "bertscore_f1": round(f1, 4),
                    "decision": self._decide(f1),
                }
            )

        return report
