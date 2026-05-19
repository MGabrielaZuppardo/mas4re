"""Unit tests for llm/json_parser.py.

Covers the failure modes observed with small models (phi3.5:3.8b, llama3.1:8b):
  1. Bare newline inside a JSON string  -> "Invalid control character"
  2. Extra text / second JSON block after closing brace -> "Extra data"
  3. Markdown code fences               -> "Expecting value"
  4. Unescaped double-quote inside string -> "Expecting ',' delimiter"
  5. Double-close pattern "word""       -> same error
  6. Combinations of the above
  7. Clean JSON must not be modified
  8. coerce_str: dict justification -> flat string
"""

from __future__ import annotations

import json

import pytest

from llm.json_parser import _next_nonws, coerce_str, extract_first_json

# ---------------------------------------------------------------------------
# _next_nonws
# ---------------------------------------------------------------------------


class TestNextNonws:
    def test_skips_spaces(self) -> None:
        assert _next_nonws("   x", 0) == (3, "x")

    def test_skips_tabs_and_newlines(self) -> None:
        assert _next_nonws(" \t\n x", 0) == (4, "x")

    def test_returns_empty_at_end(self) -> None:
        assert _next_nonws("   ", 0) == (3, "")

    def test_start_offset(self) -> None:
        assert _next_nonws("ab cd", 2) == (3, "c")


# ---------------------------------------------------------------------------
# extract_first_json — failure mode 1: bare newline inside string
# ---------------------------------------------------------------------------


class TestBareNewline:
    def test_newline_in_justification(self) -> None:
        raw = (
            '{\n  "requirement_type": "F",\n  "confidence": 0.95,\n'
            '  "classification_justification": "Funcionalidade\nque o sistema deve realizar"}'
        )
        data = json.loads(extract_first_json(raw))
        assert data["requirement_type"] == "F"
        assert "Funcionalidade" in data["classification_justification"]

    def test_carriage_return_in_value(self) -> None:
        raw = '{"key": "line1\rline2"}'
        data = json.loads(extract_first_json(raw))
        assert "line1" in data["key"]

    def test_other_control_char_dropped(self) -> None:
        raw = '{"key": "value\x07here"}'
        data = json.loads(extract_first_json(raw))
        assert data["key"] == "valuehere"


# ---------------------------------------------------------------------------
# extract_first_json — failure mode 2: extra data after }
# ---------------------------------------------------------------------------


class TestExtraData:
    def test_trailing_note(self) -> None:
        raw = (
            '{"requirement_type": "F", "confidence": 0.9, "justification": "ok"}'
            "\nNote: This is functional."
        )
        data = json.loads(extract_first_json(raw))
        assert data["requirement_type"] == "F"

    def test_trailing_blank_lines(self) -> None:
        raw = '{"priority": "M", "priority_score": 0.9}\n\n\n'
        data = json.loads(extract_first_json(raw))
        assert data["priority"] == "M"

    def test_prefix_text_before_json(self) -> None:
        raw = 'Here is the JSON:\n{"key": "value"}'
        data = json.loads(extract_first_json(raw))
        assert data["key"] == "value"


# ---------------------------------------------------------------------------
# extract_first_json — failure mode 3: markdown code fences
# ---------------------------------------------------------------------------


class TestMarkdownFences:
    def test_json_code_fence(self) -> None:
        raw = '```json\n{"requirement_type": "NF", "nfr_category": "PE"}\n```'
        data = json.loads(extract_first_json(raw))
        assert data["nfr_category"] == "PE"

    def test_plain_code_fence(self) -> None:
        raw = '```\n{"priority": "S", "priority_score": 0.8}\n```'
        data = json.loads(extract_first_json(raw))
        assert data["priority"] == "S"


# ---------------------------------------------------------------------------
# extract_first_json — failure mode 4: unescaped double-quote inside string
# ---------------------------------------------------------------------------


class TestUnescapedQuotes:
    def test_single_embedded_quoted_word(self) -> None:
        raw = (
            '{\n  "requirement_type": "F",\n  "nfr_category": null,\n  "confidence": 0.95,\n'
            '  "classification_justification": "O sistema deve realizar "X" de forma correta"}'
        )
        data = json.loads(extract_first_json(raw))
        assert data["requirement_type"] == "F"
        assert "X" in data["classification_justification"]

    def test_multiple_embedded_quotes(self) -> None:
        raw = (
            '{"requirement_type": "F", "confidence": 0.9,\n'
            '  "classification_justification": "Realiza "login" e "logout""}\nFim.'
        )
        data = json.loads(extract_first_json(raw))
        assert "login" in data["classification_justification"]
        assert "logout" in data["classification_justification"]

    def test_embedded_quote_mid_sentence(self) -> None:
        raw = (
            '{"priority": "M", "priority_score": 0.9, '
            '"justification": "Requisito "critico" para o negocio"}'
        )
        data = json.loads(extract_first_json(raw))
        assert "critico" in data["justification"]


# ---------------------------------------------------------------------------
# extract_first_json — failure mode 5: double-close pattern "word""
# ---------------------------------------------------------------------------


class TestDoubleClose:
    def test_double_close_before_brace(self) -> None:
        raw = '{"key": "value""}'
        data = json.loads(extract_first_json(raw))
        assert "value" in data["key"]

    def test_double_close_before_comma(self) -> None:
        raw = '{"key": "value"", "key2": "v2"}'
        data = json.loads(extract_first_json(raw))
        assert "value" in data["key"]
        assert data["key2"] == "v2"

    def test_double_close_real_phi35_pattern(self) -> None:
        raw = (
            '{\n  "requirement_type": "F",\n  "nfr_category": null,\n  "confidence": 0.95,\n'
            '  "classification_justification": "Este requisito descreve claramente uma'
            ' funcionalidade "especifica" que o sistema deve realizar"}'
        )
        data = json.loads(extract_first_json(raw))
        assert data["requirement_type"] == "F"
        assert data["confidence"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# extract_first_json — combinations
# ---------------------------------------------------------------------------


class TestCombinations:
    def test_newline_plus_embedded_quote(self) -> None:
        raw = (
            '{"priority": "M", "priority_score": 0.9,\n'
            '  "justification": "Requisito \\"critico\\"\\npara o sistema"}'
        )
        data = json.loads(extract_first_json(raw))
        assert data["priority"] == "M"

    def test_embedded_quote_plus_extra_data(self) -> None:
        raw = (
            '{"requirement_type": "F",\n'
            '  "classification_justification": "Funcionalidade "core" do sistema"}\n'
            "Observacao: requisito funcional confirmado."
        )
        data = json.loads(extract_first_json(raw))
        assert "core" in data["classification_justification"]

    def test_full_baseline_response_with_issues(self) -> None:
        raw = (
            '{\n  "requirement_type": "F",\n  "nfr_category": null,\n  "confidence": 0.95,\n'
            '  "classification_justification": "Este requisito "especifico" deve ser realizado",\n'
            '  "priority": "M",\n  "priority_score": 0.9,\n  "priority_rank": 1,\n'
            '  "priority_justification": "Critico para o sistema"}'
        )
        data = json.loads(extract_first_json(raw))
        assert data["priority"] == "M"
        assert data["priority_score"] == pytest.approx(0.9)
        assert data["priority_rank"] == 1


# ---------------------------------------------------------------------------
# extract_first_json — clean JSON must not be altered
# ---------------------------------------------------------------------------


class TestCleanJson:
    def test_flat_object(self) -> None:
        raw = (
            '{"requirement_type": "NF", "nfr_category": "SE",'
            ' "confidence": 0.7, "justification": "seguranca"}'
        )
        data = json.loads(extract_first_json(raw))
        assert data == {
            "requirement_type": "NF",
            "nfr_category": "SE",
            "confidence": pytest.approx(0.7),
            "justification": "seguranca",
        }

    def test_null_value_preserved(self) -> None:
        raw = '{"requirement_type": "F", "nfr_category": null, "confidence": 0.9}'
        data = json.loads(extract_first_json(raw))
        assert data["nfr_category"] is None

    def test_already_escaped_quotes_not_double_escaped(self) -> None:
        raw = '{"key": "value with \\"escaped\\" quotes"}'
        data = json.loads(extract_first_json(raw))
        assert data["key"] == 'value with "escaped" quotes'

    def test_integer_and_float_values(self) -> None:
        raw = '{"priority_rank": 1, "priority_score": 0.85}'
        data = json.loads(extract_first_json(raw))
        assert data["priority_rank"] == 1
        assert data["priority_score"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# extract_first_json — error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Nenhum objeto JSON"):
            extract_first_json("This is just plain text with no JSON.")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_first_json("")


# ---------------------------------------------------------------------------
# extract_first_json — second JSON block / balanced extraction
# ---------------------------------------------------------------------------


class TestBalancedExtraction:
    def test_stops_at_first_balanced_brace(self) -> None:
        raw = '{"requirement_type": "F", "confidence": 0.9}\n\n{"extra_block": "should be ignored"}'
        data = json.loads(extract_first_json(raw))
        assert data == {"requirement_type": "F", "confidence": pytest.approx(0.9)}

    def test_second_block_after_justification(self) -> None:
        raw = (
            '{\n  "requirement_type": "F",\n  "nfr_category": null,\n  "confidence": 0.95,\n'
            '  "classification_justification": "Funcionalidade central do sistema"\n}\n\n'
            "Justificativa adicional:\n"
            '{\n  "nota": "requisito essencial"\n}'
        )
        data = json.loads(extract_first_json(raw))
        assert data["requirement_type"] == "F"
        assert "nota" not in data

    def test_extra_data_after_prioritization(self) -> None:
        raw = (
            '{\n  "priority": "C",\n  "priority_score": 0.5,\n  "priority_rank": 2,\n'
            '  "justification": "Nao critico para funcionamento basico"\n}\n\n'
            "Justificativa: requisito de baixa prioridade."
        )
        data = json.loads(extract_first_json(raw))
        assert data["priority"] == "C"

    def test_nested_object_inside_value_not_truncated(self) -> None:
        raw = (
            '{"priority": "M", "priority_score": 1.0, "priority_rank": 1,'
            ' "justification": {"impacto": "alto", "urgencia": "media"}}'
        )
        data = json.loads(extract_first_json(raw))
        assert data["priority"] == "M"
        assert isinstance(data["justification"], dict)


# ---------------------------------------------------------------------------
# coerce_str — dict justification from phi3.5
# ---------------------------------------------------------------------------


class TestCoerceStr:
    def test_plain_string_unchanged(self) -> None:
        assert coerce_str("requisito critico") == "requisito critico"

    def test_empty_string_unchanged(self) -> None:
        assert coerce_str("") == ""

    def test_dict_flattened_to_string(self) -> None:
        d = {"impacto no negocio": "alto", "urgencia": "media"}
        result = coerce_str(d)
        assert "impacto no negocio" in result
        assert "alto" in result
        assert "urgencia" in result

    def test_dict_with_single_key(self) -> None:
        result = coerce_str({"justificativa": "essencial"})
        assert result == "justificativa: essencial"

    def test_int_converted(self) -> None:
        assert coerce_str(42) == "42"

    def test_none_converted(self) -> None:
        assert coerce_str(None) == "None"

    def test_real_phi35_justification_dict(self) -> None:
        d = {
            "impacto no negócio": "A confirmação é crucial para manter registros precisos",
            "urgência": "alta",
        }
        result = coerce_str(d)
        assert isinstance(result, str)
        assert len(result) > 0
