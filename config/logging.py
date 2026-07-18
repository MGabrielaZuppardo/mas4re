from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logging for a script entry point.

    Call this only from inside an ``if __name__ == "__main__":`` guard,
    never at module import time — scripts/run_grid.py is imported directly
    by tests/unit/test_grid_runner.py, and configuring the root logger on
    import would install handlers (and potentially create a log file)
    during pytest collection.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATEFMT, handlers=handlers)
