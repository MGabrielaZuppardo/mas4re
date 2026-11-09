"""PrioReSpa adapter and the dataset registry."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from config.settings import settings
from datasets.base import DatasetAdapter
from datasets.priorespa import PrioReSpaAdapter
from datasets.promise import PromiseAdapter
from datasets.registry import (
    DATASET_NAMES,
    build_adapter,
    default_data_lang,
    default_path,
)
from domain.enums import Lang, MoSCoWPriority, NFRCategory, RequirementType

_COLUMNS = [
    "issue_id",
    "issue_description_final",
    "f_nf_classification",
    "relevance",
    "moscow_categorization",
    "mandatory",
]


def _write(path: Path, rows: list[dict], extra: dict[str, list] | None = None) -> Path:
    frame = pd.DataFrame(rows)
    for column, values in (extra or {}).items():
        frame[column] = values
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    return path


def _row(issue_id: int, text: str, label: str = "F", moscow: int = 3, **extra) -> dict:
    return {
        "issue_id": issue_id,
        "issue_description_final": text,
        "f_nf_classification": label,
        "relevance": 3,
        "moscow_categorization": moscow,
        "mandatory": None,
        **extra,
    }


@pytest.fixture(scope="module")
def requirements():
    return PrioReSpaAdapter(settings.priorespa_dataset_path).load()


class TestRealDataset:
    def test_carrega_os_401_requisitos_com_ids_estaveis(self, requirements):
        assert len(requirements) == 401
        ids = [r.id for r in requirements]
        assert len(set(ids)) == 401
        assert "1153" in ids

    def test_mapeia_o_moscow_numerico_para_o_gabarito(self, requirements):
        counts = Counter(r.reference_priority for r in requirements)

        assert counts == {
            MoSCoWPriority.MUST_HAVE: 182,
            MoSCoWPriority.SHOULD_HAVE: 205,
            MoSCoWPriority.COULD_HAVE: 13,
            MoSCoWPriority.WONT_HAVE: 1,
        }

    def test_tipo_e_categoria_seguem_os_codigos_promise(self, requirements):
        types = Counter(r.metadata["label_type"] for r in requirements)
        categories = Counter(r.metadata["label_category"] for r in requirements)

        assert types == {"F": 315, "NF": 86}
        assert categories["US"] == 29 and categories["SE"] == 19 and categories["F"] == 315
        nfr = next(r for r in requirements if r.metadata["label_type"] == "NF")
        assert nfr.metadata["ground_truth_type"] is RequirementType.NON_FUNCTIONAL
        assert isinstance(nfr.metadata["ground_truth_category"], NFRCategory)

    def test_relevancia_obrigatoriedade_e_modulo(self, requirements):
        assert {r.metadata["ground_truth_relevance"] for r in requirements} == {1, 2, 3, 4}
        mandatory = Counter(r.metadata["mandatory"] for r in requirements)
        assert mandatory == {None: 387, True: 7, False: 7}
        assert (
            Counter(r.metadata["module"] for r in requirements)
            .most_common(1)[0][0]
            .startswith("Gesti")
        )

    def test_o_texto_e_o_original_em_espanhol_sem_copia_extra(self, requirements):
        assert all(r.source == "PRIORESPA" and r.text_en is None for r in requirements)
        assert "text_original_es" not in requirements[0].metadata


class TestSampling:
    def test_mesma_semente_mesma_amostra_e_ordem(self):
        adapter = PrioReSpaAdapter(settings.priorespa_dataset_path)

        first = [r.id for r in adapter.load_sample(20, seed=42)]
        again = [r.id for r in adapter.load_sample(20, seed=42)]
        other = [r.id for r in adapter.load_sample(20, seed=7)]

        assert first == again and first != other and len(set(first)) == 20

    def test_n_maior_que_o_dataset_devolve_tudo(self):
        adapter = PrioReSpaAdapter(settings.priorespa_dataset_path)

        assert len(adapter.load_sample(10_000, seed=1)) == 401


class TestPortugueseText:
    def test_usa_a_traducao_e_guarda_o_original(self, tmp_path):
        path = _write(
            tmp_path / "pt.csv",
            [_row(7, "[Gestión] Permitir asignar nivel.")],
            {"issue_description_pt": ["[Gestão] Permitir atribuir nível."]},
        )

        [requirement] = PrioReSpaAdapter(path, Lang.PT).load()

        assert requirement.text == "[Gestão] Permitir atribuir nível."
        assert requirement.metadata["text_original_es"] == "[Gestión] Permitir asignar nivel."
        assert requirement.metadata["module"] == "Gestión"

    def test_sem_a_coluna_traduzida_falha_com_mensagem_clara(self, tmp_path):
        path = _write(tmp_path / "es.csv", [_row(1, "Texto original suficiente.")])

        with pytest.raises(ValueError, match="issue_description_pt"):
            PrioReSpaAdapter(path, Lang.PT).load()

    def test_ingles_nao_existe_neste_dataset(self, tmp_path):
        with pytest.raises(ValueError, match="PrioReSpa"):
            PrioReSpaAdapter(tmp_path / "x.csv", Lang.EN)


class TestInvalidFiles:
    def test_arquivo_inexistente(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PrioReSpaAdapter(tmp_path / "nao_existe.csv").load()

    def test_issue_id_repetido_e_recusado(self, tmp_path):
        path = _write(tmp_path / "dup.csv", [_row(1, "Primeiro texto válido."), _row(1, "Outro.")])

        with pytest.raises(ValueError, match="único"):
            PrioReSpaAdapter(path).load()

    def test_moscow_fora_de_1_a_4_nao_e_engolido(self, tmp_path):
        path = _write(tmp_path / "bad.csv", [_row(1, "Texto válido para o teste.", moscow=9)])

        with pytest.raises(ValueError, match="ordinal"):
            PrioReSpaAdapter(path).load()

    def test_codigo_nfr_desconhecido_nao_e_engolido(self, tmp_path):
        path = _write(tmp_path / "cat.csv", [_row(1, "Texto válido para o teste.", label="ZZ")])

        with pytest.raises(ValueError):
            PrioReSpaAdapter(path).load()

    def test_obrigatoriedade_em_texto_e_lida_como_booleano(self, tmp_path):
        rows = [
            _row(1, "Texto válido número um.", mandatory="True"),
            _row(2, "Texto válido número dois.", mandatory="False"),
        ]

        first, second = PrioReSpaAdapter(_write(tmp_path / "flags.csv", rows)).load()

        assert (first.metadata["mandatory"], second.metadata["mandatory"]) == (True, False)


class TestRegistry:
    def test_conhece_os_dois_datasets(self):
        assert DATASET_NAMES == ("promise", "priorespa")

    def test_idioma_e_caminho_padrao(self):
        assert default_data_lang("promise") is Lang.PT
        assert default_data_lang("priorespa") is Lang.ES
        assert default_path("priorespa") == settings.priorespa_dataset_path
        assert default_path("priorespa", Lang.PT) == settings.priorespa_pt_dataset_path
        assert default_path("promise") == settings.promise_dataset_path

    def test_monta_o_adapter_certo(self):
        priorespa = build_adapter("priorespa")
        promise = build_adapter("promise")

        assert isinstance(priorespa, PrioReSpaAdapter) and priorespa.data_lang is Lang.ES
        assert isinstance(promise, PromiseAdapter) and isinstance(promise, DatasetAdapter)

    def test_dataset_desconhecido(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            build_adapter("nope")

    def test_promise_so_tem_texto_em_portugues(self):
        with pytest.raises(ValueError, match="PT"):
            build_adapter("promise", data_lang=Lang.ES)


class TestPromiseIdsAreStable:
    def test_ids_vem_do_indice_da_linha_nao_de_um_uuid(self, tmp_path):
        frame = pd.DataFrame(
            {
                "RequirementText": [f"English requirement number {i}" for i in range(6)],
                "RequirementText_PT": [f"Requisito em português número {i}" for i in range(6)],
                "class": ["F", "SE", "F", "PE", "F", "US"],
                "ProjectID": [1] * 6,
            }
        )
        path = tmp_path / "promise.csv"
        frame.to_csv(path, index=False)
        adapter = PromiseAdapter(path=path)

        first = [r.id for r in adapter.load()]
        again = [r.id for r in adapter.load()]
        sample = {r.text: r.id for r in adapter.load_sample(3, seed=42)}

        assert first == again == [f"promise-{i:04d}" for i in range(6)]
        assert all(first[int(text[-1])] == req_id for text, req_id in sample.items())
