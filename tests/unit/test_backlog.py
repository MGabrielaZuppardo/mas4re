from __future__ import annotations

import csv
from pathlib import Path

import pytest

from domain.enums import MoSCoWPriority, RequirementType
from domain.models import (
    ClassificationOutput,
    ClassifiedRequirement,
    PipelineState,
    PrioritizationOutput,
    PrioritizedRequirement,
    Requirement,
)
from evaluation.backlog import (
    BacklogStatus,
    backlog_from_results,
    build_backlog,
    write_backlog_csv,
    write_failed_csv,
)


def _requirement(index: int, gold_type: str | None = "F", **overrides) -> Requirement:
    metadata = {} if gold_type is None else {"label_type": gold_type, "label_category": "F"}
    fields = {
        "id": f"r{index}",
        "text": f"Requisito {index} do sistema.",
        "text_en": f"Requirement {index} of the system.",
        "metadata": metadata,
    }
    return Requirement(**{**fields, **overrides})


def _classified(
    source: Requirement,
    requirement_type: RequirementType = RequirementType.FUNCTIONAL,
    category: str | None = None,
    justification: str = "ok",
) -> ClassifiedRequirement:
    output = ClassificationOutput(
        requirement_id=source.id,
        requirement_type=requirement_type,
        nfr_category=category,
        confidence=0.9,
        justification=justification,
    )
    return ClassifiedRequirement.from_requirement(source, output)


def _prioritized(
    classified: ClassifiedRequirement,
    rank: int,
    priority: MoSCoWPriority = MoSCoWPriority.MUST_HAVE,
    justification: str = "importante",
) -> PrioritizedRequirement:
    output = PrioritizationOutput(
        requirement_id=classified.id,
        priority=priority,
        priority_score=priority.score,
        priority_rank=rank,
        justification=justification,
    )
    return PrioritizedRequirement.from_classified(classified, output)


def _detection(requirement_id: str, stage: str, severity: str, evidence: str) -> dict:
    return {
        "requirement_id": requirement_id,
        "stage": stage,
        "mode": "schema_invalid" if severity == "fatal" else "low_confidence",
        "severity": severity,
        "evidence": evidence,
    }


def _read_csv(path: Path, delimiter: str = ";") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


class TestBuildBacklog:
    def test_requisito_processado_traz_classificacao_e_prioridade(self):
        source = _requirement(1)
        classified = _classified(source, RequirementType.NON_FUNCTIONAL, "SE")
        state = PipelineState(
            raw_requirements=[source],
            classified_requirements=[classified],
            prioritized_requirements=[_prioritized(classified, rank=1)],
        )

        [row] = build_backlog(state, "run-1").rows

        assert row.status is BacklogStatus.OK
        assert (row.requirement_type, row.nfr_category) == ("NF", "SE")
        assert (row.priority, row.priority_score, row.priority_rank) == ("M", 1.0, 1)
        assert row.priority_justification == "importante"

    def test_so_classificado_fica_no_backlog_sem_prioridade(self):
        source = _requirement(1)
        state = PipelineState(
            raw_requirements=[source], classified_requirements=[_classified(source)]
        )

        backlog = build_backlog(state, "run-1")

        assert backlog.failed == []
        assert backlog.rows[0].status is BacklogStatus.CLASSIFICATION_ONLY
        assert backlog.rows[0].priority is None and backlog.rows[0].priority_rank is None

    def test_baseline_so_tem_a_lista_priorizada(self):
        source = _requirement(1)
        state = PipelineState(
            raw_requirements=[source],
            prioritized_requirements=[_prioritized(_classified(source), rank=1)],
        )

        [row] = build_backlog(state, "run-1").rows

        assert row.status is BacklogStatus.OK

    def test_requisito_sem_saida_vai_para_falhas_e_nao_para_o_backlog(self):
        processed, lost = _requirement(1), _requirement(2)
        classified = _classified(processed)
        state = PipelineState(
            raw_requirements=[processed, lost],
            classified_requirements=[classified],
            prioritized_requirements=[_prioritized(classified, rank=1)],
        )

        backlog = build_backlog(state, "run-1")

        assert [r.requirement_id for r in backlog.rows] == ["r1"]
        assert [f.requirement_id for f in backlog.failed] == ["r2"]

    def test_falha_traz_o_motivo_registrado_em_failure_detections(self):
        lost = _requirement(2)
        state = PipelineState(raw_requirements=[lost])
        state.metrics["failure_detections"] = [
            _detection("r2", "classify", "fatal", "parser fell back: no valid structured output"),
            _detection("r2", "classify", "degraded", "confidence 0.00 below threshold"),
            _detection("r9", "classify", "fatal", "de outro requisito"),
        ]

        [failed] = build_backlog(state, "run-1").failed

        assert failed.failure_stage == "classify"
        assert failed.failure_mode == "schema_invalid"
        assert failed.failure_severity == "fatal"
        assert failed.failure_evidence == "parser fell back: no valid structured output"
        assert "classify:low_confidence:degraded - confidence 0.00 below threshold" in (
            failed.detections
        )
        assert "de outro requisito" not in failed.detections

    def test_falha_sem_registro_fica_com_motivo_vazio(self):
        state = PipelineState(raw_requirements=[_requirement(1)])

        [failed] = build_backlog(state, "run-1").failed

        assert failed.failure_stage is None and failed.failure_evidence == ""
        assert failed.detections == ""

    def test_ordena_por_rank_e_deixa_sem_rank_no_fim(self):
        sources = [_requirement(i) for i in (1, 2, 3)]
        classified = [_classified(s) for s in sources]
        state = PipelineState(
            raw_requirements=sources,
            classified_requirements=classified,
            prioritized_requirements=[
                _prioritized(classified[0], rank=2),
                _prioritized(classified[2], rank=1),
            ],
        )

        rows = build_backlog(state, "run-1").rows

        assert [r.requirement_id for r in rows] == ["r3", "r1", "r2"]

    def test_gabarito_e_acerto_do_tipo(self):
        right, wrong = _requirement(1, "F"), _requirement(2, "NF")
        state = PipelineState(
            raw_requirements=[right, wrong],
            classified_requirements=[_classified(right), _classified(wrong)],
        )

        rows = {r.requirement_id: r for r in build_backlog(state, "run-1").rows}

        assert rows["r1"].gold_type == "F" and rows["r1"].type_correct is True
        assert rows["r2"].gold_type == "NF" and rows["r2"].type_correct is False

    def test_sem_gabarito_o_acerto_fica_indefinido(self):
        source = _requirement(1, gold_type=None)
        state = PipelineState(
            raw_requirements=[source], classified_requirements=[_classified(source)]
        )

        [row] = build_backlog(state, "run-1").rows

        assert row.gold_type is None and row.type_correct is None


class TestBacklogFromResults:
    def test_reconstroi_as_linhas_de_um_results_json(self):
        source = _requirement(1, "NF")
        classified = _classified(source, RequirementType.NON_FUNCTIONAL, "PE")
        results = {
            "run_id": "run-old",
            "predictions": [_prioritized(classified, rank=1).model_dump(mode="json")],
        }

        backlog = backlog_from_results(results)

        assert backlog.failed == []
        [row] = backlog.rows
        assert (row.run_id, row.nfr_category, row.priority) == ("run-old", "PE", "M")
        assert row.gold_type == "NF" and row.type_correct is True

    def test_predicao_sem_prioridade_vira_so_classificacao(self):
        classified = _classified(_requirement(1))
        results = {"run_id": "r", "predictions": [classified.model_dump(mode="json")]}

        [row] = backlog_from_results(results).rows

        assert row.status is BacklogStatus.CLASSIFICATION_ONLY


class TestWriteBacklogCsv:
    @staticmethod
    def _rows(text: str = "Requisito simples."):
        source = _requirement(1, text=text)
        state = PipelineState(
            raw_requirements=[source],
            classified_requirements=[_classified(source, justification='diz "sim"; e "não"')],
        )
        return build_backlog(state, "run-1").rows

    def test_grava_com_bom_para_o_excel_abrir_os_acentos(self, tmp_path):
        path = write_backlog_csv(self._rows("Ação do usuário."), tmp_path / "b.csv")

        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
        assert _read_csv(path)[0]["text"] == "Ação do usuário."

    def test_separador_padrao_e_ponto_e_virgula(self, tmp_path):
        path = write_backlog_csv(self._rows(), tmp_path / "b.csv")

        header = path.read_text(encoding="utf-8-sig").splitlines()[0]

        assert header.startswith("run_id;requirement_id;status;priority_rank")

    def test_separador_configuravel(self, tmp_path):
        path = write_backlog_csv(self._rows(), tmp_path / "b.csv", delimiter=",")

        assert _read_csv(path, delimiter=",")[0]["requirement_id"] == "r1"

    def test_recusa_separador_com_mais_de_um_caractere(self, tmp_path):
        with pytest.raises(ValueError):
            write_backlog_csv(self._rows(), tmp_path / "b.csv", delimiter=";;")

    def test_texto_com_separador_aspas_e_quebra_de_linha_e_preservado(self, tmp_path):
        text = 'Linha um; com ponto e vírgula\ne "aspas" na segunda.'
        path = write_backlog_csv(self._rows(text), tmp_path / "b.csv")

        row = _read_csv(path)[0]

        assert row["text"] == text
        assert row["classification_justification"] == 'diz "sim"; e "não"'

    def test_vazios_e_booleanos(self, tmp_path):
        path = write_backlog_csv(self._rows(), tmp_path / "b.csv")

        row = _read_csv(path)[0]

        assert row["priority"] == "" and row["priority_rank"] == ""
        assert row["type_correct"] == "true"

    def test_sem_a_opcao_de_gabarito_as_colunas_gold_somem(self, tmp_path):
        path = write_backlog_csv(self._rows(), tmp_path / "b.csv", include_ground_truth=False)

        columns = _read_csv(path)[0].keys()

        assert not {"gold_type", "gold_category", "type_correct"} & set(columns)

    def test_dataset_sem_gabarito_nao_gera_colunas_gold(self, tmp_path):
        source = _requirement(1, gold_type=None)
        state = PipelineState(
            raw_requirements=[source], classified_requirements=[_classified(source)]
        )

        path = write_backlog_csv(build_backlog(state, "r").rows, tmp_path / "b.csv")

        assert "gold_type" not in _read_csv(path)[0]


class TestWriteFailedCsv:
    def test_nao_cria_arquivo_quando_nao_ha_falhas(self, tmp_path):
        assert write_failed_csv([], tmp_path / "failed.csv") is None
        assert not (tmp_path / "failed.csv").exists()

    def test_grava_motivo_e_todas_as_deteccoes(self, tmp_path):
        state = PipelineState(raw_requirements=[_requirement(2)])
        state.metrics["failure_detections"] = [
            _detection("r2", "classify", "fatal", "parser fell back"),
            _detection("r2", "classify", "degraded", "baixa confiança"),
        ]

        path = write_failed_csv(build_backlog(state, "run-1").failed, tmp_path / "failed.csv")

        [row] = _read_csv(path)
        assert row["failure_stage"] == "classify"
        assert row["failure_evidence"] == "parser fell back"
        assert row["detections"] == (
            "classify:schema_invalid:fatal - parser fell back"
            " | classify:low_confidence:degraded - baixa confiança"
        )
        assert row["gold_type"] == "F"
