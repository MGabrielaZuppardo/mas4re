"""Extract the A/B run summaries that survive only in the run logs (July 2026).

The condition-B (mediated) run folders of 25-27 July 2026 were lost: their only record is the
console output kept in the `_run_*.log` files. This script parses the "Run done" block of each
log into structured JSON, so the numbers no longer depend on grepping logs.

Usage (from the repository root):
    python -m experiments.extract_ab_runs_from_logs

Output:
    experiments/results/archive_pre_adr011_20260920/ab_runs_from_logs.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from experiments.paths import HISTORICAL_RESULTS_DIR

LOG_GLOB = "_run_*.log"
OUTPUT_NAME = "ab_runs_from_logs.json"
_ENCODINGS = ("utf-8", "cp1252", "utf-16")
_RUN_DONE = re.compile(r"Run done \| strategy=(\S+) \| n=(\S+)")
_FILENAME = re.compile(
    r"_run_(?P<condition>[ab])_(?P<lang>pt|en)(?:_(?P<model>qwen|llama))?(?:_(?P<rep>rep\d+))?"
)
_PARSE_FAILURE_LINE = "JSON decode falhou"


def read_log(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _flatten(text: str) -> str:
    return re.sub(r"\s*\n\s*", " ", text.replace("\r", ""))


def _extract_dict(flat: str, key: str) -> dict[str, Any] | None:
    """Parse the dict literal that follows `key:` (balanced braces, nested dicts allowed)."""
    start = flat.find(f"{key}: {{")
    if start < 0:
        return None
    begin = flat.index("{", start)
    depth = 0
    for position in range(begin, len(flat)):
        depth += {"{": 1, "}": -1}.get(flat[position], 0)
        if depth == 0:
            return ast.literal_eval(flat[begin : position + 1])
    return None


def parse_run_log(text: str) -> dict[str, Any]:
    flat = _flatten(text)
    summary: dict[str, Any] = {
        "complete": False,
        "parse_failures_logged": text.count(_PARSE_FAILURE_LINE),
    }
    done = _RUN_DONE.search(flat)
    if done is None:
        return summary
    summary.update(
        complete=True,
        strategy=done.group(1),
        n=done.group(2),
        classification=_extract_dict(flat, "classification"),
        moscow=_extract_dict(flat, "moscow"),
        mediation=_extract_dict(flat, "mediation"),
    )
    return summary


def describe_filename(name: str) -> dict[str, str | None]:
    match = _FILENAME.match(name)
    if match is None:
        return {"condition": None, "lang": None, "model_hint": None, "rep": None}
    return {
        "condition": match["condition"].upper(),
        "lang": match["lang"],
        "model_hint": match["model"],
        "rep": match["rep"],
    }


def extract_runs(log_dir: Path) -> list[dict[str, Any]]:
    return [
        {"log": path.name, **describe_filename(path.name), **parse_run_log(read_log(path))}
        for path in sorted(log_dir.glob(LOG_GLOB))
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", type=Path, default=HISTORICAL_RESULTS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    runs = extract_runs(args.log_dir)
    output = args.log_dir / OUTPUT_NAME
    output.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    complete = sum(1 for run in runs if run["complete"])
    print(f"{len(runs)} logs, {complete} complete -> {output}")


if __name__ == "__main__":
    main()
