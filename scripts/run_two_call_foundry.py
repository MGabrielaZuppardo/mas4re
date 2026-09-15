"""Executa apenas o braço two_call_baseline para foundry/gpt-4.1-mini (PT e EN),
o braço que faltava para completar a comparação Δp/Δs suplementar (fora do
grid de 18 condições) discutida na resposta ao Comentário 3 (Revisor 3).

Reaproveita ExperimentRunner/TwoCallBaselineStrategy, mesma config dos runs
de baseline/pipeline já existentes (n=625, seed=42, temperature=0.0).

Uso (a partir de D:/mas4re):
    python scripts/run_two_call_foundry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from domain.enums import Lang
from experiments.runner import ExperimentRunner, RunConfig
from experiments.strategy import TwoCallBaselineStrategy

MODEL = "foundry/gpt-4.1-mini"


def main() -> None:
    runner = ExperimentRunner()
    for lang in ["pt", "en"]:
        print(f"\n{'=' * 72}\ntwo_call_baseline | {MODEL} | {lang}\n{'=' * 72}")
        strategy = TwoCallBaselineStrategy(model=MODEL, lang=Lang(lang))
        config = RunConfig(
            strategy_name="two_call_baseline",
            model=MODEL,
            dataset_path=settings.promise_dataset_path,
            lang=lang,
            n_samples=None,
            seed=42,
        )
        result = runner.execute(strategy, config)
        print(f"Concluído: {lang} -> run_id={result.run_id} elapsed={result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
