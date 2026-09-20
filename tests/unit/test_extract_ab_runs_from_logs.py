from pathlib import Path

from experiments.extract_ab_runs_from_logs import (
    describe_filename,
    extract_runs,
    parse_run_log,
)

_COMPLETE_LOG = """\
LangChainPendingDeprecationWarning: something
JSON decode falhou | req_id=abc | erro=x
JSON decode falhou | req_id=def | erro=y
Run done | strategy=mediated_pipeline | n=full
  classification: {'accuracy': 0.808, 'f1_macro': 0.8065, 'f1_weighted':
0.8096, 'mcc': 0.6282, 'n_evaluated': 625}
  moscow: {'M': 0.3936, 'S': 0.56, 'C': 0.0432, 'W': 0.0032}
  mediation: {'n_passes': 2, 'n_conflicts_detected': 2,
'n_requirements_resent': 2, 'n_conflicts_resolved_after_retry': 0,
'conflicts_by_category': {'SE': 2}}
"""


class TestParseRunLog:
    def test_le_metricas_de_um_log_completo(self):
        summary = parse_run_log(_COMPLETE_LOG)

        assert summary["complete"] is True
        assert summary["strategy"] == "mediated_pipeline"
        assert summary["n"] == "full"
        assert summary["classification"]["accuracy"] == 0.808
        assert summary["classification"]["f1_weighted"] == 0.8096
        assert summary["moscow"]["M"] == 0.3936

    def test_le_o_dicionario_aninhado_da_mediacao(self):
        mediation = parse_run_log(_COMPLETE_LOG)["mediation"]

        assert mediation["n_conflicts_detected"] == 2
        assert mediation["n_conflicts_resolved_after_retry"] == 0
        assert mediation["conflicts_by_category"] == {"SE": 2}

    def test_conta_as_falhas_de_parse_registradas(self):
        assert parse_run_log(_COMPLETE_LOG)["parse_failures_logged"] == 2

    def test_log_sem_run_done_fica_incompleto(self):
        summary = parse_run_log("apenas avisos, a execucao nao terminou")

        assert summary["complete"] is False
        assert "classification" not in summary

    def test_pipeline_sem_bloco_de_mediacao(self):
        log = "Run done | strategy=pipeline | n=full\n  classification: {'accuracy': 0.87}\n"

        summary = parse_run_log(log)

        assert summary["mediation"] is None
        assert summary["moscow"] is None


class TestDescribeFilename:
    def test_condicao_idioma_modelo_e_repeticao(self):
        assert describe_filename("_run_b_pt_qwen_rep2.log") == {
            "condition": "B",
            "lang": "pt",
            "model_hint": "qwen",
            "rep": "rep2",
        }

    def test_nome_sem_modelo_nem_repeticao(self):
        described = describe_filename("_run_a_en.log")

        assert described["condition"] == "A" and described["model_hint"] is None

    def test_nome_desconhecido(self):
        assert describe_filename("outro.log")["condition"] is None


class TestExtractRuns:
    def test_lista_apenas_logs_run_em_ordem(self, tmp_path: Path):
        (tmp_path / "_run_b_pt_qwen.log").write_text(_COMPLETE_LOG, encoding="utf-8")
        (tmp_path / "_run_a_pt_qwen.log").write_text("sem resultado", encoding="utf-8")
        (tmp_path / "grid_run.log").write_text(_COMPLETE_LOG, encoding="utf-8")

        runs = extract_runs(tmp_path)

        assert [r["log"] for r in runs] == ["_run_a_pt_qwen.log", "_run_b_pt_qwen.log"]
        assert [r["complete"] for r in runs] == [False, True]
        assert runs[1]["condition"] == "B"

    def test_decodifica_log_em_cp1252(self, tmp_path: Path):
        (tmp_path / "_run_b_pt_qwen.log").write_bytes(
            ("conteúdo com acento\n" + _COMPLETE_LOG).encode("cp1252")
        )

        assert extract_runs(tmp_path)[0]["complete"] is True
