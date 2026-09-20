"""Smoke tests for the mas4re CLI (no real LLM/dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import typer.main
from typer.testing import CliRunner

from cli.main import app
from domain.enums import MoSCoWPriority, RequirementType
from domain.models import (
    ClassificationOutput,
    ClassifiedRequirement,
    PrioritizationOutput,
    PrioritizedRequirement,
    Requirement,
)

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


def _run_dir_with_predictions(tmp_path: Path, n_configured: int = 2) -> Path:
    predictions = []
    for rank in (1, 2):
        source = Requirement(
            id=f"r{rank}",
            text=f"Requisito {rank} do sistema.",
            metadata={"label_type": "F", "label_category": "F"},
        )
        classified = ClassifiedRequirement.from_requirement(
            source,
            ClassificationOutput(
                requirement_id=source.id,
                requirement_type=RequirementType.FUNCTIONAL,
                confidence=0.9,
                justification="ok",
            ),
        )
        prioritized = PrioritizedRequirement.from_classified(
            classified,
            PrioritizationOutput(
                requirement_id=source.id,
                priority=MoSCoWPriority.MUST_HAVE,
                priority_score=1.0,
                priority_rank=rank,
                justification="ok",
            ),
        )
        predictions.append(prioritized.model_dump(mode="json"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps({"run_id": "run-x", "config": {"n": n_configured}, "predictions": predictions}),
        encoding="utf-8",
    )
    return run_dir


def test_cli_backlog_gera_o_csv_a_partir_do_results(tmp_path: Path) -> None:
    run_dir = _run_dir_with_predictions(tmp_path)

    result = runner.invoke(app, ["backlog", str(run_dir)])

    assert result.exit_code == 0
    assert "2 requirements" in result.output
    lines = (run_dir / "backlog.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 3 and lines[0].startswith("run_id;requirement_id")


def test_cli_backlog_avisa_quando_faltam_requisitos_no_results(tmp_path: Path) -> None:
    run_dir = _run_dir_with_predictions(tmp_path, n_configured=5)

    result = runner.invoke(app, ["backlog", str(run_dir)])

    assert result.exit_code == 0
    assert "3 requirements have no output" in result.output


def test_cli_backlog_aceita_saida_separador_e_sem_gabarito(tmp_path: Path) -> None:
    run_dir = _run_dir_with_predictions(tmp_path)
    target = tmp_path / "saida" / "meu_backlog.csv"

    result = runner.invoke(
        app,
        ["backlog", str(run_dir), "--out", str(target), "--delimiter", ",", "--no-ground-truth"],
    )

    assert result.exit_code == 0
    header = target.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.startswith("run_id,requirement_id") and "gold_type" not in header


def test_cli_backlog_sem_results_json_falha(tmp_path: Path) -> None:
    assert runner.invoke(app, ["backlog", str(tmp_path)]).exit_code != 0


def test_cli_backlog_recusa_separador_invalido(tmp_path: Path) -> None:
    run_dir = _run_dir_with_predictions(tmp_path)

    result = runner.invoke(app, ["backlog", str(run_dir), "--delimiter", ";;"])

    assert result.exit_code != 0


def test_cli_run_expoe_as_opcoes_do_backlog() -> None:
    params = {p.name for p in typer.main.get_command(app).commands["run"].params}

    assert {"backlog_delimiter", "no_ground_truth"} <= params
