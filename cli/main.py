from __future__ import annotations

import json
from pathlib import Path

import typer
from click.core import ParameterSource
from rich.console import Console
from rich.table import Table

from agents.agentic.runtime import DEFAULT_MAX_STEPS
from config.settings import settings
from datasets.registry import (
    DATASET_NAMES,
    PROMISE,
    build_adapter,
    default_data_lang,
    default_path,
)
from domain.enums import Lang
from evaluation.backlog import (
    BACKLOG_FILENAME,
    DEFAULT_DELIMITER,
    backlog_from_results,
    write_backlog_csv,
)
from experiments.analyze_protocols import find_arms
from experiments.confirmatory import ARMS, arm_run_values
from experiments.exploratory import (
    RESULTS_DIR,
    TRACES_DIR,
    load_exploratory_spec,
    manifest_block,
    variant_run_values,
    warmup_seed_from_run,
)
from experiments.runner import ExperimentRunner, RunConfig, check_tag
from experiments.strategy import (
    AGENTIC_STRATEGY_PROTOCOLS,
    AgenticPipelineStrategy,
    BaselineStrategy,
    MediatedPipelineStrategy,
    PipelineStrategy,
)
from llm.preflight import preflight
from pipeline.mediation import MUST_SHARE_LIMIT

app = typer.Typer(
    add_completion=False,
    help="MAS4RE -- multi-agent vs single-agent requirements pipeline.",
)
console = Console()


_STRATEGIES = {"baseline", "pipeline", "pipeline_mediated", *AGENTIC_STRATEGY_PROTOCOLS}

_ARM_LOCKED_OPTIONS = (
    "strategy",
    "model",
    "pri_model",
    "n",
    "seed",
    "dataset_name",
    "dataset",
    "data_lang",
    "lang",
    "temperature",
    "max_steps",
    "tag",
)


def _reject_overrides_of_the_design(ctx: typer.Context, arm: str) -> None:
    given = [
        name.replace("_", "-")
        for name in _ARM_LOCKED_OPTIONS
        if ctx.get_parameter_source(name) is not ParameterSource.DEFAULT
    ]
    if given:
        raise typer.BadParameter(
            f"--arm takes everything from confirmatory.yml; remove --{', --'.join(given)} "
            f"(or drop --arm for an exploratory run)"
        )
    if arm not in ARMS:
        raise typer.BadParameter(f"--arm must be one of {list(ARMS)}")


def _check_models_work(models: list[str], needs_tools: bool) -> None:
    for model in dict.fromkeys(models):
        result = preflight(model)
        console.print(f"  preflight: {result.describe()}")
        if not result.ok or (needs_tools and not result.tool_calling):
            raise typer.Exit(code=1)


@app.command()
def explore(
    variant: str | None = typer.Option(None, help="Exploratory variant (see --list)."),
    arm: str | None = typer.Option(None, help="Arm of the variant to run."),
    list_variants: bool = typer.Option(False, "--list", help="Show the variants and their arms."),
    n: int | None = typer.Option(None, help="Sample size, for a smoke test (default: all)."),
    no_preflight: bool = typer.Option(False, "--no-preflight", help="Skip the model check."),
) -> None:
    """Run one arm of an EXPLORATORY variant (ADR-016). Never confirmatory: the results go to
    experiments/exploratory/ and the manifest says so."""
    spec = load_exploratory_spec()
    if list_variants:
        for name, declared in spec["variants"].items():
            console.print(f"{name}: arms {declared['arms']} | {declared['question']}")
        return
    if variant is None or arm is None:
        raise typer.BadParameter("give --variant and --arm (or --list)")
    try:
        values = variant_run_values(variant, arm)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if not no_preflight:
        _check_models_work([values["model"], values["pri_model"]], needs_tools=True)
    data_language = Lang(values["data_lang"])
    dataset_path = default_path(values["dataset_name"], data_language)
    seed, pinned = None, None
    if values["pin_warmup_from"]:
        reference = find_arms(Path("experiments") / "results")[values["pin_warmup_from"]]
        requirements = build_adapter(values["dataset_name"], dataset_path, data_language).load()
        seed = warmup_seed_from_run(reference, requirements, spec["warmup_size"])
        pinned = reference.name
    strategy = AgenticPipelineStrategy(
        classifier_model=values["model"],
        prioritizer_model=values["pri_model"],
        temperature=values["temperature"],
        lang=Lang(values["lang"]),
        max_steps=values["max_steps"],
        protocol=AGENTIC_STRATEGY_PROTOCOLS[values["strategy"]],
        must_limit=values["must_limit"] or MUST_SHARE_LIMIT,
        prioritizer_warmup_seed=seed,
    )
    config = RunConfig(
        strategy_name=values["strategy"],
        model=f"{values['model']}+{values['pri_model']}",
        dataset_path=dataset_path,
        lang=values["lang"],
        n_samples=n,
        seed=values["seed"],
        temperature=values["temperature"],
        dataset_name=values["dataset_name"],
        data_lang=values["data_lang"],
        tag=values["tag"],
        max_steps=values["max_steps"],
        exploratory=manifest_block(variant, arm, pinned),
    )
    result = ExperimentRunner(out_dir=str(RESULTS_DIR), trace_dir=str(TRACES_DIR)).execute(
        strategy, config
    )
    console.print(f"[bold yellow]Exploratory run done[/] | {variant} / {arm} | {result.run_id}")


@app.command(name="check-model")
def check_model(model: str = typer.Argument(..., help="e.g. ollama/qwen2.5:7b")) -> None:
    """Build a model and make two tiny calls (plain and tool calling) to see it can run."""
    result = preflight(model)
    console.print(result.describe())
    if not result.ok or not result.tool_calling:
        raise typer.Exit(code=1)


@app.command()
def run(
    ctx: typer.Context,
    strategy: str | None = typer.Option(
        None,
        help="baseline | pipeline | pipeline_mediated | "
        "pipeline_agentic (P0) | pipeline_agentic_p2 | pipeline_agentic_p3",
    ),
    arm: str | None = typer.Option(
        None,
        help="p0 | p0-repeat | p2 | p3 | control: run that arm exactly as frozen in "
        "experiments/configs/confirmatory.yml (model, languages, seed and tag included).",
    ),
    no_preflight: bool = typer.Option(
        False, "--no-preflight", help="Skip the check that the model(s) can run."
    ),
    model: str = typer.Option(settings.classifier_model, help="Model for baseline / classifier."),
    pri_model: str = typer.Option(
        settings.prioritizer_model, help="Prioritizer model (pipeline only)."
    ),
    n: int | None = typer.Option(None, help="Sample size (None = full dataset)."),
    seed: int = typer.Option(42, help="Sampling seed."),
    dataset_name: str = typer.Option(PROMISE, "--dataset-name", help="promise | priorespa"),
    dataset: str | None = typer.Option(
        None, help="Dataset CSV path (default: the dataset's own file)."
    ),
    data_lang: str | None = typer.Option(
        None,
        "--data-lang",
        help="Language of the requirement text: pt | es (default: the dataset's original).",
    ),
    lang: str = typer.Option("pt", help="Language of the prompts: pt | en | es"),
    temperature: float = typer.Option(0.0, help="LLM temperature."),
    max_steps: int = typer.Option(
        DEFAULT_MAX_STEPS, help="Step budget of each agent episode (pipeline_agentic* only)."
    ),
    backlog_delimiter: str = typer.Option(
        DEFAULT_DELIMITER, help="CSV delimiter of backlog.csv / failed.csv."
    ),
    no_reference_labels: bool = typer.Option(
        False,
        "--no-reference-labels",
        help="Leave the reference_* columns out of backlog.csv / failed.csv.",
    ),
    tag: str | None = typer.Option(
        None,
        help="Label kept in the manifest and the folder name, e.g. p0-repeat (noise floor).",
    ),
) -> None:
    """Run one strategy under a frozen config and persist artifacts."""
    if arm:
        _reject_overrides_of_the_design(ctx, arm)
        values = arm_run_values(arm)
        strategy, model, pri_model = values["strategy"], values["model"], values["pri_model"]
        lang, data_lang, dataset_name = values["lang"], values["data_lang"], values["dataset_name"]
        seed, temperature, n, tag, max_steps = (
            values[k] for k in ("seed", "temperature", "n", "tag", "max_steps")
        )
    elif strategy is None:
        raise typer.BadParameter("give --strategy (exploratory) or --arm (the frozen design)")
    if strategy not in _STRATEGIES:
        raise typer.BadParameter(f"strategy must be one of {sorted(_STRATEGIES)}")
    _check_delimiter(backlog_delimiter)
    try:
        check_tag(tag)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if dataset_name not in DATASET_NAMES:
        raise typer.BadParameter(f"dataset-name must be one of {list(DATASET_NAMES)}")
    lang_enum = Lang(lang)
    data_language = Lang(data_lang) if data_lang else default_data_lang(dataset_name)
    if not no_preflight:
        models = [model] if strategy == "baseline" else [model, pri_model]
        _check_models_work(models, needs_tools=strategy != "baseline")
    if strategy == "baseline":
        strat: (
            BaselineStrategy | PipelineStrategy | MediatedPipelineStrategy | AgenticPipelineStrategy
        ) = BaselineStrategy(model=model, temperature=temperature, lang=lang_enum)
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
        strat = AgenticPipelineStrategy(
            classifier_model=model,
            prioritizer_model=pri_model,
            temperature=temperature,
            lang=lang_enum,
            max_steps=max_steps,
            protocol=AGENTIC_STRATEGY_PROTOCOLS[strategy],
        )
        run_model = f"{model}+{pri_model}"

    config = RunConfig(
        strategy_name=strategy,
        model=run_model,
        dataset_path=dataset or default_path(dataset_name, data_language),
        lang=lang,
        n_samples=n,
        seed=seed,
        temperature=temperature,
        dataset_name=dataset_name,
        data_lang=data_language.value,
        tag=tag,
        max_steps=max_steps if strategy in AGENTIC_STRATEGY_PROTOCOLS else None,
    )
    result = ExperimentRunner(
        backlog_delimiter=backlog_delimiter, include_reference=not no_reference_labels
    ).execute(strat, config)

    console.print(f"[bold green]Run done[/] | strategy={strategy} | n={config.n_samples or 'full'}")
    console.print(f"  backlog: {result.backlog_path}")
    if result.failed_path:
        console.print(f"  [yellow]failed requirements:[/] {result.failed_path}")
    if result.actions_path:
        console.print(f"  actions: {result.actions_path}")
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
    no_reference_labels: bool = typer.Option(
        False, "--no-reference-labels", help="Leave the reference_* columns out."
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
        rows,
        Path(out) if out else Path(run_dir) / BACKLOG_FILENAME,
        delimiter,
        not no_reference_labels,
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
