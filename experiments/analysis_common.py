"""Shared helpers for the analysis and diagnostic scripts in experiments/.

Analysis scripts read the historical full-dataset runs (results.json + manifest.json)
and write a JSON with a manifest under experiments/results/analysis/, so every
reported number is traceable to a commit, a dataset hash and a set of parameters.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from config.settings import settings
from datasets.promise import PromiseAdapter
from domain.enums import InformationRegime
from domain.models import Requirement
from evaluation.cross_agent_check import _CRITICAL_NFR, _LOW_PRIORITIES
from experiments.paths import ACTIVE_RESULTS_DIR, HISTORICAL_RESULTS_DIR

Prediction = dict[str, Any]

ANALYSIS_DIR = ACTIVE_RESULTS_DIR / "analysis"
FULL_DATASET_N = 625
DEFAULT_SEED = 42
_ENCODINGS = ("utf-8", "cp1252")
PARSE_FAILURE_PREFIX = "Parse falhou"
_TRACKED_LIBRARIES = ("numpy", "scikit-learn", "pandas", "scipy", "langgraph")


class MixedInformationRegimeError(ValueError):
    """Raised when one analysis would combine runs from different information regimes."""


@dataclass(frozen=True)
class RunData:
    strategy: str
    model: str
    lang: str | None
    timestamp: str
    predictions: list[Prediction]
    metrics: dict[str, Any]
    path: str
    information_regime: InformationRegime = InformationRegime.ZERO_SHOT


def read_json(path: str | Path) -> Any:
    raw = Path(path).read_bytes()
    for encoding in _ENCODINGS:
        try:
            return json.loads(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path} as any of {_ENCODINGS}")


def short_model_name(model: str) -> str:
    """'ollama/qwen2.5:7b-ollama/qwen2.5:7b' or 'a+a' -> the single model name."""
    first = model.split("+")[0].split("-ollama/")[0]
    return first.removeprefix("ollama/")


@dataclass(frozen=True)
class RunSource:
    """A directory holding run folders (each with results.json + manifest.json)."""

    root: Path
    recursive: bool

    def result_files(self) -> list[str]:
        pattern = "**/results.json" if self.recursive else "*/results.json"
        return sorted(glob.glob((self.root / pattern).as_posix(), recursive=self.recursive))


HISTORICAL = "historical"
ACTIVE = "active"

# A generation is one coherent version of the agents, so runs from different
# generations are never analysed together by accident. "active" is depth-1 only:
# archives nested under experiments/results/ do not leak into it.
GENERATIONS: dict[str, tuple[RunSource, ...]] = {
    HISTORICAL: (
        RunSource(HISTORICAL_RESULTS_DIR, recursive=False),
        RunSource(Path("results"), recursive=True),
    ),
    ACTIVE: (RunSource(ACTIVE_RESULTS_DIR, recursive=False),),
}


def ensure_single_regime(runs: list[RunData]) -> None:
    regimes = sorted({run.information_regime.value for run in runs})
    if len(regimes) > 1:
        raise MixedInformationRegimeError(
            f"Refusing to analyse runs from different information regimes together: {regimes}"
        )


def load_full_runs(
    strategy: str | None = None,
    n: int = FULL_DATASET_N,
    generation: str = HISTORICAL,
    allow_mixed_regimes: bool = False,
) -> list[RunData]:
    """Runs of one generation. Manifests without `information_regime` (all runs before it
    existed) are zero-shot by definition."""
    runs: list[RunData] = []
    for source in GENERATIONS[generation]:
        for path in source.result_files():
            data = read_json(path)
            config = data.get("config", {})
            if config.get("n") != n or (strategy and config.get("strategy") != strategy):
                continue
            manifest = read_json(Path(path).with_name("manifest.json"))
            runs.append(
                RunData(
                    strategy=config["strategy"],
                    model=short_model_name(manifest["model"]),
                    lang=manifest.get("lang"),
                    timestamp=manifest.get("timestamp_utc", ""),
                    predictions=data["predictions"],
                    metrics=data.get("metrics", {}),
                    path=Path(path).as_posix(),
                    information_regime=InformationRegime(
                        manifest.get("information_regime", InformationRegime.ZERO_SHOT)
                    ),
                )
            )
    if not allow_mixed_regimes:
        ensure_single_regime(runs)
    return runs


def latest_run_per_model_and_lang(runs: list[RunData]) -> dict[str, dict[str, RunData]]:
    """{lang: {model: run}}, keeping the most recent run of each (model, lang)."""
    latest: dict[str, dict[str, RunData]] = defaultdict(dict)
    for run in runs:
        if not run.lang:
            continue
        current = latest[run.lang].get(run.model)
        if current is None or run.timestamp > current.timestamp:
            latest[run.lang][run.model] = run
    return {lang: dict(sorted(models.items())) for lang, models in sorted(latest.items())}


def load_promise_sample(n: int, seed: int = DEFAULT_SEED) -> list[Requirement]:
    return PromiseAdapter(path=settings.promise_dataset_path).load_sample(n, seed=seed)


def is_parse_failure(pred: Prediction) -> bool:
    return str(pred.get("justification", "")).startswith(PARSE_FAILURE_PREFIX)


def is_type_error(pred: Prediction) -> bool:
    return bool(pred["requirement_type"] != pred["metadata"]["label_type"])


def has_structural_inconsistency(pred: Prediction) -> bool:
    is_nfr = pred["requirement_type"] == "NF"
    return is_nfr != (pred.get("nfr_category") is not None)


def is_critical_nfr(pred: Prediction) -> bool:
    return pred.get("nfr_category") in _CRITICAL_NFR


def has_inter_agent_conflict(pred: Prediction) -> bool:
    return is_critical_nfr(pred) and pred.get("priority") in _LOW_PRIORITIES


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half_width, center + half_width


def required_sample_size(proportion: float, half_width: float, z: float = 1.96) -> int:
    return math.ceil(z * z * proportion * (1 - proportion) / half_width**2)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else "unknown"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def build_analysis_manifest(
    script: str, information_regime: InformationRegime, parameters: dict[str, Any]
) -> dict[str, Any]:
    return {
        "script": script,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "information_regime": information_regime.value,
        "python": sys.version.split()[0],
        "libraries": _library_versions(),
        "dataset": settings.promise_dataset_path,
        "dataset_md5": _file_md5(Path(settings.promise_dataset_path)),
        "parameters": parameters,
    }


def write_analysis_result(
    name: str,
    manifest: dict[str, Any],
    results: dict[str, Any],
    out_dir: Path = ANALYSIS_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{name}_{stamp}.json"
    payload = {"manifest": manifest, "results": results}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"
