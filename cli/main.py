from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from config.settings import settings
from domain.enums import Lang
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import BaselineStrategy, PipelineStrategy

app = typer.Typer(
    add_completion=False,
    help="MAS4RE -- multi-agent vs single-agent requirements pipeline.",
)
console = Console()


@app.command()
def run(
    strategy: str = typer.Option(..., help="baseline | pipeline"),
    model: str = typer.Option(settings.classifier_model, help="Model for baseline / classifier."),
    pri_model: str = typer.Option(
        settings.prioritizer_model, help="Prioritizer model (pipeline only)."
    ),
    n: int | None = typer.Option(None, help="Sample size (None = full dataset)."),
    seed: int = typer.Option(42, help="Sampling seed."),
    dataset: str = typer.Option(settings.promise_dataset_path, help="Dataset CSV path."),
    lang: str = typer.Option("pt", help="pt | en"),
    temperature: float = typer.Option(0.0, help="LLM temperature."),
) -> None:
    """Run one strategy under a frozen config and persist artifacts."""
    lang_enum = Lang(lang)
    if strategy == "baseline":
        strat: BaselineStrategy | PipelineStrategy = BaselineStrategy(
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
    else:
        raise typer.BadParameter("strategy must be 'baseline' or 'pipeline'")

    config = RunConfig(
        strategy_name=strategy,
        model=run_model,
        dataset_path=dataset,
        n_samples=n,
        seed=seed,
        temperature=temperature,
    )
    result = ExperimentRunner().execute(strat, config)

    console.print(f"[bold green]Run done[/] | strategy={strategy} | n={config.n_samples or 'full'}")
    cls = result.metrics.get("classification", {})
    if cls:
        console.print(f"  classification: {cls}")
    if "moscow_distribution" in result.metrics:
        console.print(f"  moscow: {result.metrics['moscow_distribution']}")


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
