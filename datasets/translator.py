"""Translates one column of a dataset with an LLM, item by item, resumably.

Every translated item is appended to a checkpoint file as soon as it is done, so a run can stop at
any moment (a command must stay under ten minutes) and the next one continues where it stopped.
The translated file is written only when every item has a translation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

logger = logging.getLogger(__name__)

_PROMPT = (
    "Translate the following software issue from {source_name} to {target_name}.\n"
    "Rules:\n"
    "- Return ONLY the translation: no comments, no quotes around it, no preamble.\n"
    "- Leave exactly as written anything that is code, an identifier (camelCase, snake_case, "
    "kebab-case), a file name, a URL, a number, or something between backticks.\n"
    "- Keep the structure: the leading tag in square brackets (translate the word inside it), "
    "line breaks and lists.\n"
    "- Do not summarize, explain or add anything.\n\n"
    "Text:\n{text}"
)

_PREAMBLE = re.compile(
    r"^\s*(?:aqui\s+est[áa]\s+a\s+tradu[çc][ãa]o|tradu[çc][ãa]o)\s*:\s*", re.IGNORECASE
)
_FENCE = re.compile(r"^```[a-z]*\n?|\n?```$")
_NUMBERS = re.compile(r"\d+")
_MIN_LENGTH_RATIO = 0.4
_MAX_LENGTH_RATIO = 2.5


class _Message(Protocol):
    content: Any


class ChatModel(Protocol):
    def invoke(self, prompt: str) -> _Message: ...


@dataclass(frozen=True)
class TranslationJob:
    id_column: str
    source_column: str
    translated_column: str
    source_name: str = "Spanish"
    target_name: str = "Brazilian Portuguese"
    separator: str = ";"
    encoding: str = "utf-8-sig"


@dataclass(frozen=True)
class Progress:
    translated_now: int
    done: int
    total: int
    flagged: int

    @property
    def remaining(self) -> int:
        return self.total - self.done

    @property
    def complete(self) -> bool:
        return self.remaining == 0


def clean_translation(raw: str) -> str:
    """Drop what models add around a translation: code fences and a 'Tradução:' preamble."""
    text = _FENCE.sub("", raw.strip()).strip()
    return _PREAMBLE.sub("", text).strip()


def translation_issues(original: str, translated: str) -> list[str]:
    """Cheap checks that catch the usual failures: empty output, a summary or a runaway
    answer, a lost leading tag, and numbers that changed."""
    if not translated.strip():
        return ["empty"]
    issues = []
    ratio = len(translated) / max(len(original), 1)
    if not _MIN_LENGTH_RATIO <= ratio <= _MAX_LENGTH_RATIO:
        issues.append(f"length_ratio={ratio:.2f}")
    if original.lstrip().startswith("[") != translated.lstrip().startswith("["):
        issues.append("leading_tag")
    if sorted(_NUMBERS.findall(original)) != sorted(_NUMBERS.findall(translated)):
        issues.append("numbers")
    return issues


def read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    entries = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {entry["id"]: entry for entry in entries}


class DatasetTranslator:
    def __init__(self, llm: ChatModel, job: TranslationJob, model: str) -> None:
        self._llm = llm
        self._job = job
        self._model = model

    def _frame(self, input_path: Path) -> pd.DataFrame:
        job = self._job
        return pd.read_csv(
            input_path, sep=job.separator, encoding=job.encoding, dtype={job.id_column: str}
        )

    def _translate(self, text: str) -> str:
        prompt = _PROMPT.format(
            source_name=self._job.source_name, target_name=self._job.target_name, text=text
        )
        return clean_translation(str(self._llm.invoke(prompt).content))

    def _translate_checked(self, text: str) -> tuple[str, list[str]]:
        """One retry when the first answer fails a check; a still-failing answer is kept and
        flagged, so a person reviews it instead of the run stalling."""
        translated = self._translate(text)
        issues = translation_issues(text, translated)
        if issues:
            retry = self._translate(text)
            retry_issues = translation_issues(text, retry)
            if len(retry_issues) <= len(issues):
                translated, issues = retry, retry_issues
        return translated, issues

    def run(
        self,
        input_path: Path,
        checkpoint_path: Path,
        limit: int,
        max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> Progress:
        """Translate up to `limit` pending items, stopping early after `max_seconds`."""
        job = self._job
        frame = self._frame(input_path)
        done = read_checkpoint(checkpoint_path)
        pending = [row for _, row in frame.iterrows() if str(row[job.id_column]) not in done]
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        started = clock()
        translated_now = 0
        with checkpoint_path.open("a", encoding="utf-8") as handle:
            for row in pending[:limit]:
                if clock() - started >= max_seconds:
                    break
                item_id = str(row[job.id_column])
                text = str(row[job.source_column])
                translated, issues = self._translate_checked(text)
                entry = {
                    "id": item_id,
                    "text_pt": translated,
                    "model": self._model,
                    "flags": issues,
                }
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                handle.flush()
                done[item_id] = entry
                translated_now += 1
                if issues:
                    logger.warning("Tradução suspeita | id=%s | %s", item_id, issues)
        flagged = sum(1 for entry in done.values() if entry["flags"])
        return Progress(translated_now, len(done), len(frame), flagged)

    def finalize(self, input_path: Path, checkpoint_path: Path, output_path: Path) -> Path:
        """Write the dataset with the translated column added, once nothing is pending."""
        job = self._job
        frame = self._frame(input_path)
        done = read_checkpoint(checkpoint_path)
        missing = [i for i in frame[job.id_column] if str(i) not in done]
        if missing:
            raise ValueError(f"{len(missing)} item(s) not translated yet, e.g. {missing[:3]}")
        frame[job.translated_column] = [done[str(i)]["text_pt"] for i in frame[job.id_column]]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, sep=job.separator, encoding=job.encoding, index=False)
        return output_path
