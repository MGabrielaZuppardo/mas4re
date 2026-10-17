"""Resumable LLM translation of a dataset column."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from datasets.priorespa import PrioReSpaAdapter
from datasets.translator import (
    DatasetTranslator,
    TranslationJob,
    clean_translation,
    read_checkpoint,
    translation_issues,
)
from domain.enums import Lang

_JOB = TranslationJob(
    id_column="issue_id",
    source_column="issue_description_final",
    translated_column="issue_description_pt",
)


class ScriptedTranslator:
    """Answers with `translate(text)` and remembers every prompt it was shown."""

    def __init__(self, translate=lambda text: f"{text} [pt]") -> None:
        self.prompts: list[str] = []
        self._translate = translate

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        text = prompt.split("Text:\n", 1)[1]
        return SimpleNamespace(content=self._translate(text))


def _dataset(tmp_path: Path, n: int = 4) -> Path:
    path = tmp_path / "es.csv"
    pd.DataFrame(
        {
            "issue_id": [f"{100 + i}" for i in range(n)],
            "issue_description_final": [
                f"[Gestión] Requisito número {i} del sistema" for i in range(n)
            ],
            "f_nf_classification": ["F"] * n,
            "relevance": [3] * n,
            "moscow_categorization": [3] * n,
            "mandatory": [None] * n,
        }
    ).to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    return path


def _translator(llm) -> DatasetTranslator:
    return DatasetTranslator(llm, _JOB, model="fake/model")


class TestCleaning:
    def test_tira_preambulo_e_cerca_de_codigo(self):
        assert clean_translation("Tradução: [Gestão] Texto") == "[Gestão] Texto"
        assert clean_translation("```\n[Gestão] Texto\n```") == "[Gestão] Texto"
        assert clean_translation("Aqui está a tradução: Olá") == "Olá"

    def test_deixa_intacto_o_que_ja_esta_limpo(self):
        assert clean_translation("  [Gestão] Permitir atribuir nível  ") == (
            "[Gestão] Permitir atribuir nível"
        )


class TestChecks:
    def test_traducao_boa_nao_tem_problemas(self):
        assert translation_issues("[Gestión] Hay 3 usuarios", "[Gestão] Há 3 usuários") == []

    def test_vazia(self):
        assert translation_issues("texto", "   ") == ["empty"]

    def test_tamanho_incompativel_indica_resumo_ou_divagacao(self):
        issues = translation_issues("a" * 200, "curto")

        assert any(issue.startswith("length_ratio") for issue in issues)

    def test_perda_do_marcador_inicial_e_mudanca_de_numeros(self):
        issues = translation_issues("[UI] Cambiar 2 botones", "Mudar 3 botões")

        assert "leading_tag" in issues and "numbers" in issues


class TestRun:
    def test_traduz_e_grava_o_ponto_de_retomada_item_a_item(self, tmp_path):
        llm = ScriptedTranslator()
        checkpoint = tmp_path / "ckpt.jsonl"

        progress = _translator(llm).run(_dataset(tmp_path), checkpoint, limit=10, max_seconds=60)

        assert (progress.translated_now, progress.done, progress.total) == (4, 4, 4)
        assert progress.complete and progress.remaining == 0
        saved = read_checkpoint(checkpoint)
        assert set(saved) == {"100", "101", "102", "103"}
        assert saved["100"]["text_pt"] == "[Gestión] Requisito número 0 del sistema [pt]"
        assert saved["100"]["model"] == "fake/model"

    def test_limite_para_no_numero_pedido_e_o_proximo_comando_continua(self, tmp_path):
        dataset, checkpoint = _dataset(tmp_path), tmp_path / "ckpt.jsonl"
        translator = _translator(ScriptedTranslator())

        first = translator.run(dataset, checkpoint, limit=3, max_seconds=60)
        second = translator.run(dataset, checkpoint, limit=3, max_seconds=60)

        assert (first.translated_now, first.remaining) == (3, 1)
        assert (second.translated_now, second.complete) == (1, True)
        assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 4

    def test_nao_traduz_de_novo_o_que_ja_esta_no_ponto_de_retomada(self, tmp_path):
        dataset, checkpoint = _dataset(tmp_path), tmp_path / "ckpt.jsonl"
        _translator(ScriptedTranslator()).run(dataset, checkpoint, limit=2, max_seconds=60)
        llm = ScriptedTranslator()

        _translator(llm).run(dataset, checkpoint, limit=10, max_seconds=60)

        assert len(llm.prompts) == 2

    def test_orcamento_de_tempo_interrompe_antes_do_limite(self, tmp_path):
        ticks = iter([0.0, 1.0, 2.0, 9.0, 99.0, 99.0, 99.0])
        progress = _translator(ScriptedTranslator()).run(
            _dataset(tmp_path),
            tmp_path / "ckpt.jsonl",
            limit=10,
            max_seconds=5,
            clock=lambda: next(ticks),
        )

        assert progress.translated_now == 2 and not progress.complete

    def test_resposta_ruim_tenta_uma_segunda_vez_e_fica_com_a_melhor(self, tmp_path):
        answers = iter(["", "[Gestão] Requisito número 0 do sistema"])
        llm = ScriptedTranslator(lambda text: next(answers))

        progress = _translator(llm).run(
            _dataset(tmp_path, n=1), tmp_path / "ckpt.jsonl", limit=1, max_seconds=60
        )

        saved = read_checkpoint(tmp_path / "ckpt.jsonl")["100"]
        assert len(llm.prompts) == 2 and saved["flags"] == [] and progress.flagged == 0

    def test_resposta_que_continua_ruim_e_mantida_mas_sinalizada_para_revisao(self, tmp_path):
        llm = ScriptedTranslator(lambda text: "x")

        progress = _translator(llm).run(
            _dataset(tmp_path, n=1), tmp_path / "ckpt.jsonl", limit=1, max_seconds=60
        )

        assert progress.flagged == 1
        assert read_checkpoint(tmp_path / "ckpt.jsonl")["100"]["flags"]

    def test_o_prompt_pede_so_a_traducao_e_protege_codigo(self, tmp_path):
        llm = ScriptedTranslator()

        _translator(llm).run(_dataset(tmp_path, n=1), tmp_path / "c.jsonl", limit=1, max_seconds=60)

        prompt = llm.prompts[0]
        assert "from Spanish to Brazilian Portuguese" in prompt and "ONLY the translation" in prompt
        assert "identifier" in prompt and "[Gestión] Requisito número 0" in prompt


class TestFinalize:
    def test_escreve_o_dataset_com_a_coluna_traduzida_e_o_adapter_le(self, tmp_path):
        dataset, checkpoint, output = (
            _dataset(tmp_path),
            tmp_path / "ckpt.jsonl",
            tmp_path / "pt.csv",
        )
        translator = _translator(
            ScriptedTranslator(lambda text: text.replace("Requisito", "Requisito PT"))
        )
        translator.run(dataset, checkpoint, limit=10, max_seconds=60)

        translator.finalize(dataset, checkpoint, output)

        requirements = PrioReSpaAdapter(output, Lang.PT).load()
        assert [r.id for r in requirements] == ["100", "101", "102", "103"]
        assert "Requisito PT número 0" in requirements[0].text
        assert requirements[0].metadata["text_original_es"].startswith("[Gestión] Requisito")
        assert "issue_description_final" in pd.read_csv(output, sep=";", encoding="utf-8-sig")

    def test_recusa_escrever_com_itens_pendentes(self, tmp_path):
        dataset, checkpoint = _dataset(tmp_path), tmp_path / "ckpt.jsonl"
        translator = _translator(ScriptedTranslator())
        translator.run(dataset, checkpoint, limit=2, max_seconds=60)

        with pytest.raises(ValueError, match="not translated yet"):
            translator.finalize(dataset, checkpoint, tmp_path / "pt.csv")

    def test_ponto_de_retomada_e_jsonl_valido_linha_a_linha(self, tmp_path):
        checkpoint = tmp_path / "ckpt.jsonl"
        _translator(ScriptedTranslator()).run(
            _dataset(tmp_path), checkpoint, limit=4, max_seconds=60
        )

        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == {"id", "text_pt", "model", "flags"}
