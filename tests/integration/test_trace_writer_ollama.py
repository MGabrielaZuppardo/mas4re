"""Integration tests for TraceWriter com Ollama real (PR #22).

Requerem Ollama rodando localmente com qwen2.5:7b e llama3.1:8b.
NÃO são executados no CI (apenas tests/unit/ entra no pipeline).

Executar localmente:
    pytest tests/integration/test_trace_writer_ollama.py -v --tb=short

O que é validado:
- ClassificationAgent e BaselineAgent com trace_writer ativo geram
  um arquivo JSONL válido em experiments/traces/
- Cada requisito produz exatamente um evento
- Campos obrigatórios estão presentes e com valores plausíveis
- latency_ms reflete tempo real de chamada ao LLM (> 100 ms)
- parsed_ok=True na maioria dos itens (modelo respondendo bem)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.baseline import BaselineAgent
from agents.classifier import ClassificationAgent
from config.settings import settings
from datasets.promise import PromiseAdapter
from evaluation.trace_writer import TraceWriter

SAMPLE_N = 5
TRACE_DIR = Path("experiments") / "traces" / "test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def promise_sample():
    adapter = PromiseAdapter(path=settings.promise_dataset_path)
    return adapter.load_sample(n=SAMPLE_N, seed=42)


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    return tmp_path / "traces"


# ---------------------------------------------------------------------------
# ClassificationAgent + TraceWriter
# ---------------------------------------------------------------------------


class TestClassificationAgentTrace:
    def test_gera_jsonl_com_n_linhas(self, promise_sample, trace_dir) -> None:
        """Um evento por requisito classificado."""
        run_id = "integ-classify-01"
        with TraceWriter(output_dir=trace_dir, run_id=run_id) as tw:
            agent = ClassificationAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent.classify_batch(promise_sample, max_workers=2)

        lines = (trace_dir / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == SAMPLE_N

    def test_campos_obrigatorios_presentes(self, promise_sample, trace_dir) -> None:
        run_id = "integ-classify-02"
        with TraceWriter(output_dir=trace_dir, run_id=run_id) as tw:
            agent = ClassificationAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent.classify_batch(promise_sample, max_workers=2)

        required = {
            "run_id",
            "stage",
            "requirement_id",
            "model",
            "latency_ms",
            "parsed_ok",
            "timestamp",
        }
        for line in (trace_dir / f"{run_id}.jsonl").read_text().splitlines():
            payload = json.loads(line)
            missing = required - payload.keys()
            assert not missing, f"Campos ausentes: {missing}"

    def test_stage_e_model_corretos(self, promise_sample, trace_dir) -> None:
        run_id = "integ-classify-03"
        with TraceWriter(output_dir=trace_dir, run_id=run_id) as tw:
            agent = ClassificationAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent.classify_batch(promise_sample, max_workers=2)

        for line in (trace_dir / f"{run_id}.jsonl").read_text().splitlines():
            payload = json.loads(line)
            assert payload["stage"] == "classify"
            assert payload["model"] == settings.classifier_model

    def test_latency_ms_positiva_e_plausivel(self, promise_sample, trace_dir) -> None:
        """LLM local leva > 100 ms por requisito em hardware típico."""
        run_id = "integ-classify-04"
        with TraceWriter(output_dir=trace_dir, run_id=run_id) as tw:
            agent = ClassificationAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent.classify_batch(promise_sample, max_workers=2)

        latencies = [
            json.loads(raw)["latency_ms"]
            for raw in (trace_dir / f"{run_id}.jsonl").read_text().splitlines()
        ]
        assert all(ms > 0 for ms in latencies), "Latência deve ser positiva"
        assert any(ms > 100 for ms in latencies), "Ao menos um evento deve ter latência > 100 ms"

    def test_maioria_parsed_ok(self, promise_sample, trace_dir) -> None:
        """Modelo respondendo bem: ≥ 80 % dos eventos com parsed_ok=True."""
        run_id = "integ-classify-05"
        with TraceWriter(output_dir=trace_dir, run_id=run_id) as tw:
            agent = ClassificationAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent.classify_batch(promise_sample, max_workers=2)

        events = [
            json.loads(raw) for raw in (trace_dir / f"{run_id}.jsonl").read_text().splitlines()
        ]
        ok_rate = sum(1 for e in events if e["parsed_ok"]) / len(events)
        print(f"\nparsed_ok rate: {ok_rate:.0%}")
        assert ok_rate >= 0.8, f"parsed_ok rate abaixo de 80 %: {ok_rate:.0%}"

    def test_requirement_ids_cobrem_sample(self, promise_sample, trace_dir) -> None:
        """Os IDs rastreados correspondem exatamente aos requisitos do batch."""
        run_id = "integ-classify-06"
        with TraceWriter(output_dir=trace_dir, run_id=run_id) as tw:
            agent = ClassificationAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent.classify_batch(promise_sample, max_workers=2)

        traced_ids = {
            json.loads(raw)["requirement_id"]
            for raw in (trace_dir / f"{run_id}.jsonl").read_text().splitlines()
        }
        sample_ids = {r.id for r in promise_sample}
        assert traced_ids == sample_ids

    def test_jsonl_cada_linha_valida(self, promise_sample, trace_dir) -> None:
        run_id = "integ-classify-07"
        with TraceWriter(output_dir=trace_dir, run_id=run_id) as tw:
            agent = ClassificationAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent.classify_batch(promise_sample, max_workers=2)

        for line in (trace_dir / f"{run_id}.jsonl").read_text().splitlines():
            json.loads(line)  # raises se inválido


# ---------------------------------------------------------------------------
# BaselineAgent + TraceWriter
# ---------------------------------------------------------------------------


class TestBaselineAgentTrace:
    def test_gera_jsonl_stage_baseline(self, promise_sample, trace_dir) -> None:
        run_id = "integ-baseline-01"
        with TraceWriter(output_dir=trace_dir, run_id=run_id) as tw:
            agent = BaselineAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent._run_batch(promise_sample, max_workers=2)

        lines = (trace_dir / f"{run_id}.jsonl").read_text().splitlines()
        assert len(lines) == SAMPLE_N
        for line in lines:
            assert json.loads(line)["stage"] == "baseline"

    def test_latencia_baseline_positiva(self, promise_sample, trace_dir) -> None:
        run_id = "integ-baseline-02"
        with TraceWriter(output_dir=trace_dir, run_id=run_id) as tw:
            agent = BaselineAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent._run_batch(promise_sample, max_workers=2)

        latencies = [
            json.loads(raw)["latency_ms"]
            for raw in (trace_dir / f"{run_id}.jsonl").read_text().splitlines()
        ]
        assert all(ms > 0 for ms in latencies)

    def test_run_id_consistente_no_arquivo(self, promise_sample, trace_dir) -> None:
        run_id = "integ-baseline-03"
        with TraceWriter(output_dir=trace_dir, run_id=run_id) as tw:
            agent = BaselineAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent._run_batch(promise_sample, max_workers=2)

        for line in (trace_dir / f"{run_id}.jsonl").read_text().splitlines():
            assert json.loads(line)["run_id"] == run_id


# ---------------------------------------------------------------------------
# Comparação pipeline vs. baseline — latência média
# ---------------------------------------------------------------------------


class TestLatenciaComparativa:
    def test_imprime_latencia_media_por_strategy(self, promise_sample, trace_dir) -> None:
        """Não é assertion — imprime dados para análise SQ2 (custo computacional)."""
        classify_id = "integ-compare-classify"
        baseline_id = "integ-compare-baseline"

        with TraceWriter(output_dir=trace_dir, run_id=classify_id) as tw:
            agent = ClassificationAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent.classify_batch(promise_sample, max_workers=2)

        with TraceWriter(output_dir=trace_dir, run_id=baseline_id) as tw:
            agent = BaselineAgent(
                model=settings.classifier_model,
                trace_writer=tw,
            )
            agent._run_batch(promise_sample, max_workers=2)

        def _avg_latency(run: str) -> float:
            lines = (trace_dir / f"{run}.jsonl").read_text().splitlines()
            return sum(json.loads(raw)["latency_ms"] for raw in lines) / len(lines)

        avg_classify = _avg_latency(classify_id)
        avg_baseline = _avg_latency(baseline_id)

        print(f"\nLatência média classify : {avg_classify:.1f} ms")
        print(f"Latência média baseline : {avg_baseline:.1f} ms")
        print(f"Razão classify/baseline : {avg_classify / avg_baseline:.2f}x")

        assert avg_classify > 0
        assert avg_baseline > 0
