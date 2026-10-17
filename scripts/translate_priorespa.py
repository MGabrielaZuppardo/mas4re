"""Translate PrioReSpa from Spanish to Brazilian Portuguese with an LLM, in resumable batches.

Each command translates until `--limit` items or `--max-seconds` is reached (keep it under ten
minutes) and records progress in a checkpoint; run it again to continue. When every item is done
the translated dataset is written to `--output`.

Usage:
    python scripts/translate_priorespa.py --limit 25                 # first batch, for a look
    python scripts/translate_priorespa.py                            # next batch (resumes)
    python scripts/translate_priorespa.py --model foundry/gpt-4.1-mini

Then validate it (back-translation + BERTScore):
    python scripts/validate_translation.py --pair priorespa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from datasets.translator import DatasetTranslator, TranslationJob
from llm.factory import build_llm

_DEFAULT_CHECKPOINT = Path("datasets/data/priorespa/translation_checkpoint.jsonl")
_JOB = TranslationJob(
    id_column="issue_id",
    source_column="issue_description_final",
    translated_column="issue_description_pt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate PrioReSpa ES -> PT-BR")
    parser.add_argument("--model", default="ollama/qwen2.5:7b", help="LLM that translates")
    parser.add_argument("--input", default=settings.priorespa_dataset_path)
    parser.add_argument("--output", default=settings.priorespa_pt_dataset_path)
    parser.add_argument("--checkpoint", default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--limit", type=int, default=100, help="Max items in this run")
    parser.add_argument(
        "--max-seconds", type=float, default=480.0, help="Stop starting new items after this"
    )
    args = parser.parse_args()

    translator = DatasetTranslator(build_llm(args.model, 0.0), _JOB, args.model)
    progress = translator.run(Path(args.input), Path(args.checkpoint), args.limit, args.max_seconds)
    print(
        f"translated now: {progress.translated_now} | done: {progress.done}/{progress.total} "
        f"| flagged for review: {progress.flagged}"
    )
    if progress.complete:
        written = translator.finalize(Path(args.input), Path(args.checkpoint), Path(args.output))
        print(f"complete -> {written}")
    else:
        print(f"{progress.remaining} left: run the command again to continue")


if __name__ == "__main__":
    main()
