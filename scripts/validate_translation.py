"""Validate the PROMISE-PT dataset translation quality.

Uses back-translation (PT -> EN via Ollama) + BERTScore to assess
semantic preservation without human reference translations.

Uso:
    python scripts/validate_translation.py
    python scripts/validate_translation.py --model ollama/qwen2.5:7b
    python scripts/validate_translation.py --threshold-accept 0.88
    python scripts/validate_translation.py --output results/validation.csv

Artefatos gerados:
    datasets/data/promise_nfr/validation_report.csv   (scores por requisito)
    datasets/data/promise_nfr/validation_summary.json (estatísticas agregadas)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from datasets.translation_validator import TranslationValidator

_DEFAULT_OUTPUT = Path("datasets/data/promise_nfr/validation_report.csv")
_DEFAULT_SUMMARY = Path("datasets/data/promise_nfr/validation_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PROMISE-PT translation via back-translation + BERTScore"
    )
    parser.add_argument(
        "--dataset",
        default=settings.promise_dataset_path,
        help="Path to the PROMISE-PT CSV file",
    )
    parser.add_argument(
        "--model",
        default="ollama/qwen2.5:7b",
        help="Ollama model for back-translation (PT -> EN)",
    )
    parser.add_argument(
        "--threshold-accept",
        type=float,
        default=0.90,
        help="Min BERTScore F1 to accept a translation (default: 0.90)",
    )
    parser.add_argument(
        "--threshold-review",
        type=float,
        default=0.80,
        help="Min BERTScore F1 to flag for review (default: 0.80)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="BERTScore batch size (default: 32)",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="Output CSV path for per-requirement scores",
    )
    parser.add_argument(
        "--summary",
        default=str(_DEFAULT_SUMMARY),
        help="Output JSON path for aggregate summary",
    )
    args = parser.parse_args()

    print("\nMAS4RE Translation Validator")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Model    : {args.model}")
    print(f"  Thresholds: accept >= {args.threshold_accept} | review >= {args.threshold_review}\n")

    validator = TranslationValidator(
        model=args.model,
        batch_size=args.batch_size,
        threshold_accept=args.threshold_accept,
        threshold_review=args.threshold_review,
    )

    report = validator.validate(args.dataset)

    # Save per-requirement CSV
    report.save(args.output)

    # Save aggregate JSON summary
    summary = report.summary()
    out_summary = Path(args.summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved to: {out_summary}")

    # Print to console
    report.print_summary()

    # Exit code: 1 if any rejected translations
    if report.n_rejected > 0:
        print(f"\nWARNING: {report.n_rejected} requirement(s) rejected — check {args.output}")
        sys.exit(1)


if __name__ == "__main__":
    main()
