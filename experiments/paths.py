"""Locations of experiment artifacts.

ACTIVE_RESULTS_DIR receives new runs. HISTORICAL_RESULTS_DIR holds the runs behind the
submitted paper's numbers, produced by single-call agents before ADR-011 and the critique fix
(see its ARCHIVE.md). Paths are relative to the repository root.
"""

from __future__ import annotations

from pathlib import Path

ACTIVE_RESULTS_DIR = Path("experiments") / "results"
HISTORICAL_RESULTS_DIR = ACTIVE_RESULTS_DIR / "archive_pre_adr011_20260920"
