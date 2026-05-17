"""Smoke tests for the mas4re CLI (no real LLM/dataset)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "compare" in result.output


def test_cli_run_invalid_strategy() -> None:
    result = runner.invoke(app, ["run", "--strategy", "bogus", "--n", "1"])
    assert result.exit_code != 0


def test_cli_eval_reads_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps({"metrics": {"classification": {"f1_macro": 0.8}}})
    )
    result = runner.invoke(app, ["eval", str(run_dir)])
    assert result.exit_code == 0
    assert "f1_macro" in result.output


def test_cli_compare(tmp_path: Path) -> None:
    for name in ("b", "p"):
        d = tmp_path / name
        d.mkdir()
        (d / "results.json").write_text(
            json.dumps({"metrics": {"classification": {"f1_macro": 0.7}}})
        )
    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            str(tmp_path / "b"),
            "--pipeline",
            str(tmp_path / "p"),
        ],
    )
    assert result.exit_code == 0
    assert "Baseline" in result.output
