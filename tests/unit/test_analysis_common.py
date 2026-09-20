import json
from pathlib import Path

import pytest

from experiments.analysis_common import (
    HYBRID_EXPLORATORY,
    ZERO_SHOT,
    build_analysis_manifest,
    has_inter_agent_conflict,
    has_structural_inconsistency,
    is_critical_nfr,
    is_parse_failure,
    is_type_error,
    latest_run_per_model_and_lang,
    load_full_runs,
    ratio,
    read_json,
    required_sample_size,
    short_model_name,
    wilson_interval,
    write_analysis_result,
)


def _prediction(**overrides):
    base = {
        "requirement_type": "F",
        "nfr_category": None,
        "confidence": 0.9,
        "priority": "M",
        "justification": "ok",
        "metadata": {"label_type": "F"},
    }
    return {**base, **overrides}


def _write_run(
    root: Path,
    name: str,
    *,
    strategy="pipeline",
    n=625,
    model="ollama/qwen2.5:7b",
    lang="pt",
    timestamp="2026-01-01T00:00:00+00:00",
    preds=None,
    base="experiments/results",
) -> None:
    run_dir = root / base / name
    run_dir.mkdir(parents=True)
    results = {"config": {"strategy": strategy, "n": n}, "predictions": preds or [], "metrics": {}}
    manifest = {"model": model, "lang": lang, "timestamp_utc": timestamp}
    (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestReadJson:
    def test_le_utf8(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_text('{"texto": "usuário"}', encoding="utf-8")
        assert read_json(path) == {"texto": "usuário"}

    def test_cai_para_cp1252(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_bytes('{"texto": "usuário"}'.encode("cp1252"))
        assert read_json(path) == {"texto": "usuário"}

    def test_recusa_bytes_indecodificaveis(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_bytes(b'{"texto": "\x81"}')
        with pytest.raises(ValueError):
            read_json(path)


class TestShortModelName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ollama/qwen2.5:7b", "qwen2.5:7b"),
            ("ollama/llama3.1:8b-ollama/llama3.1:8b", "llama3.1:8b"),
            ("foundry/gpt-4.1-mini+foundry/gpt-4.1-mini", "foundry/gpt-4.1-mini"),
        ],
    )
    def test_normaliza(self, raw, expected):
        assert short_model_name(raw) == expected


class TestPredicates:
    def test_falha_de_parse(self):
        assert is_parse_failure(_prediction(justification="Parse falhou (JSON): x"))
        assert not is_parse_failure(_prediction())

    def test_erro_de_tipo(self):
        assert is_type_error(_prediction(requirement_type="NF"))
        assert not is_type_error(_prediction())

    @pytest.mark.parametrize(
        ("requirement_type", "category", "inconsistent"),
        [("NF", None, True), ("NF", "SE", False), ("F", "SE", True), ("F", None, False)],
    )
    def test_inconsistencia_estrutural(self, requirement_type, category, inconsistent):
        pred = _prediction(requirement_type=requirement_type, nfr_category=category)
        assert has_structural_inconsistency(pred) is inconsistent

    @pytest.mark.parametrize(
        ("category", "priority", "conflict"),
        [
            ("SE", "C", True),
            ("PE", "W", True),
            ("SE", "M", False),
            ("US", "C", False),
            ("SE", None, False),
        ],
    )
    def test_conflito_inter_agentes(self, category, priority, conflict):
        pred = _prediction(nfr_category=category, priority=priority)
        assert has_inter_agent_conflict(pred) is conflict

    def test_categoria_critica(self):
        assert is_critical_nfr(_prediction(nfr_category="FT"))
        assert not is_critical_nfr(_prediction(nfr_category="LF"))


class TestStatistics:
    def test_ratio_com_denominador_zero_devolve_none(self):
        assert ratio(1, 0) is None
        assert ratio(1, 4) == 0.25

    def test_wilson_13_de_15(self):
        low, high = wilson_interval(13, 15)
        assert low == pytest.approx(0.621, abs=1e-3)
        assert high == pytest.approx(0.963, abs=1e-3)

    def test_wilson_sem_amostra_e_intervalo_total(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_tamanho_de_amostra_necessario(self):
        assert required_sample_size(0.87, 0.10) == 44
        assert required_sample_size(0.87, 0.05) == 174


class TestLoadFullRuns:
    def test_filtra_por_n_e_estrategia(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_run(tmp_path, "keep")
        _write_run(tmp_path, "small", n=4)
        _write_run(tmp_path, "other", strategy="baseline")

        pipeline = load_full_runs(strategy="pipeline", generation="active")
        everything = load_full_runs(generation="active")

        assert [r.strategy for r in pipeline] == ["pipeline"]
        assert len(everything) == 2

    def test_normaliza_o_nome_do_modelo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_run(tmp_path, "a", model="ollama/qwen2.5:7b-ollama/qwen2.5:7b")

        assert load_full_runs(generation="active")[0].model == "qwen2.5:7b"

    def test_ultima_execucao_por_modelo_e_idioma(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_run(tmp_path, "old", timestamp="2026-01-01T00:00:00+00:00")
        _write_run(tmp_path, "new", timestamp="2026-02-01T00:00:00+00:00")
        _write_run(tmp_path, "nolang", lang=None)

        latest = latest_run_per_model_and_lang(load_full_runs(generation="active"))

        assert list(latest) == ["pt"]
        assert latest["pt"]["qwen2.5:7b"].timestamp.startswith("2026-02")

    def test_geracao_ativa_nao_enxerga_o_arquivo_historico(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_run(tmp_path, "new")
        _write_run(tmp_path, "archive_pre_adr011_20260920/old")

        runs = load_full_runs(generation="active")

        assert [Path(r.path).parent.name for r in runs] == ["new"]

    def test_geracao_historica_le_o_arquivo_e_a_pasta_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_run(tmp_path, "new")
        _write_run(tmp_path, "archive_pre_adr011_20260920/old")
        _write_run(tmp_path, "archive_v1/legacy", base="results")

        runs = load_full_runs(generation="historical")

        assert sorted(Path(r.path).parent.name for r in runs) == ["legacy", "old"]


class TestManifestAndOutput:
    def test_manifest_registra_regime_e_parametros(self):
        manifest = build_analysis_manifest("s", ZERO_SHOT, {"n": 15})

        assert manifest["information_regime"] == "zero_shot"
        assert manifest["parameters"] == {"n": 15}
        assert {"git_commit", "timestamp_utc", "python", "libraries", "dataset_md5"} <= set(
            manifest
        )

    def test_regime_hibrido_e_distinto_do_zero_shot(self):
        assert HYBRID_EXPLORATORY != ZERO_SHOT

    def test_write_analysis_result_grava_manifest_e_resultados(self, tmp_path):
        manifest = build_analysis_manifest("s", ZERO_SHOT, {})

        path = write_analysis_result("demo", manifest, {"x": 1}, out_dir=tmp_path / "out")

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert path.name.startswith("demo_") and path.suffix == ".json"
        assert payload["results"] == {"x": 1}
        assert payload["manifest"]["script"] == "s"
