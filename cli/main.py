from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from config.settings import settings
from domain.enums import Lang
from evaluation.backlog import (
    BACKLOG_FILENAME,
    DEFAULT_DELIMITER,
    backlog_from_results,
    write_backlog_csv,
)
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import BaselineStrategy, MediatedPipelineStrategy, PipelineStrategy

app = typer.Typer(
    add_completion=False,
    help="MAS4RE -- multi-agent vs single-agent requirements pipeline.",
)
console = Console()


@app.command()
def run(
    strategy: str = typer.Option(..., help="baseline | pipeline | pipeline_mediated"),
    model: str = typer.Option(settings.classifier_model, help="Model for baseline / classifier."),
    pri_model: str = typer.Option(
        settings.prioritizer_model, help="Prioritizer model (pipeline only)."
    ),
    n: int | None = typer.Option(None, help="Sample size (None = full dataset)."),
    seed: int = typer.Option(42, help="Sampling seed."),
    dataset: str = typer.Option(settings.promise_dataset_path, help="Dataset CSV path."),
    lang: str = typer.Option("pt", help="pt | en"),
    temperature: float = typer.Option(0.0, help="LLM temperature."),
    backlog_delimiter: str = typer.Option(
        DEFAULT_DELIMITER, help="CSV delimiter of backlog.csv / failed.csv."
    ),
    no_ground_truth: bool = typer.Option(
        False, "--no-ground-truth", help="Leave the gold_* columns out of backlog.csv / failed.csv."
    ),
) -> None:
    """Run one strategy under a frozen config and persist artifacts."""
    _check_delimiter(backlog_delimiter)
    lang_enum = Lang(lang)
    if strategy == "baseline":
        strat: BaselineStrategy | PipelineStrategy | MediatedPipelineStrategy = BaselineStrategy(
            model=model, temperature=temperature, lang=lang_enum
        )
        run_model = model
    elif strategy == "pipeline":
        strat = PipelineStrategy(
            classifier_model=model,
            prioritizer_model=pri_model,
            temperature=temperature,
            lang=lang_enum,
        )
        run_model = f"{model}+{pri_model}"
    elif strategy == "pipeline_mediated":
        strat = MediatedPipelineStrategy(
            classifier_model=model,
            prioritizer_model=pri_model,
            temperature=temperature,
            lang=lang_enum,
        )
        run_model = f"{model}+{pri_model}"
    else:
        raise typer.BadParameter("strategy must be 'baseline', 'pipeline' or 'pipeline_mediated'")

    config = RunConfig(
        strategy_name=strategy,
        model=run_model,
        dataset_path=dataset,
        lang=lang,
        n_samples=n,
        seed=seed,
        temperature=temperature,
    )
    result = ExperimentRunner(
        backlog_delimiter=backlog_delimiter, include_ground_truth=not no_ground_truth
    ).execute(strat, config)

    console.print(f"[bold green]Run done[/] | strategy={strategy} | n={config.n_samples or 'full'}")
    console.print(f"  backlog: {result.backlog_path}")
    if result.failed_path:
        console.print(f"  [yellow]failed requirements:[/] {result.failed_path}")
    cls = result.metrics.get("classification", {})
    if cls:
        console.print(f"  classification: {cls}")
    if "moscow_distribution" in result.metrics:
        console.print(f"  moscow: {result.metrics['moscow_distribution']}")


def _check_delimiter(delimiter: str) -> None:
    if len(delimiter) != 1:
        raise typer.BadParameter(f"delimiter must be a single character, got {delimiter!r}")


@app.command()
def backlog(
    run_dir: str = typer.Argument(..., help="Run directory with results.json"),
    delimiter: str = typer.Option(DEFAULT_DELIMITER, help="CSV delimiter."),
    no_ground_truth: bool = typer.Option(
        False, "--no-ground-truth", help="Leave the gold_* columns out."
    ),
    out: str | None = typer.Option(None, help="Output CSV (default: <run_dir>/backlog.csv)."),
) -> None:
    """Rebuild backlog.csv (classification + priority) from a finished run's results.json."""
    _check_delimiter(delimiter)
    results_path = Path(run_dir) / "results.json"
    if not results_path.exists():
        raise typer.BadParameter(f"results.json not found in {run_dir}")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    rows = backlog_from_results(results).rows
    target = write_backlog_csv(
        rows, Path(out) if out else Path(run_dir) / BACKLOG_FILENAME, delimiter, not no_ground_truth
    )

    console.print(f"[bold green]Backlog written[/] | {len(rows)} requirements | {target}")
    n_missing = results.get("config", {}).get("n", len(rows)) - len(rows)
    if n_missing > 0:
        console.print(
            f"  [yellow]{n_missing} requirements have no output[/] (results.json does not "
            "store which ones)."
        )


@app.command(name="eval")
def eval_run(
    run_dir: str = typer.Argument(..., help="Run directory with results.json"),
) -> None:
    """Print metrics from a previous run's results.json."""
    path = Path(run_dir) / "results.json"
    if not path.exists():
        raise typer.BadParameter(f"results.json not found in {run_dir}")
    data = json.loads(path.read_text())
    console.print_json(json.dumps(data.get("metrics", {})))


@app.command()
def compare(
    baseline: str = typer.Option(..., help="Baseline run directory."),
    pipeline: str = typer.Option(..., help="Pipeline run directory."),
) -> None:
    """Side-by-side classification metrics: baseline vs pipeline."""

    def _load(d: str) -> dict:
        return json.loads((Path(d) / "results.json").read_text())

    b = _load(baseline)["metrics"].get("classification", {})
    p = _load(pipeline)["metrics"].get("classification", {})

    table = Table(title="Baseline vs Pipeline -- Classification")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Pipeline", justify="right")
    for key in sorted(set(b) | set(p)):
        table.add_row(key, str(b.get(key, "--")), str(p.get(key, "--")))
    console.print(table)


if __name__ == "__main__":
    app()
