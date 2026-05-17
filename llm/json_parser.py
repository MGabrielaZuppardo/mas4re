"""Robust JSON extraction for LLM responses.

Small models (phi3.5, llama3.1, etc.) often produce two failure modes:
  - Extra text after the closing brace ("Extra data" from json.loads)
  - Unescaped control characters inside string values ("Invalid control
    character" from json.loads, e.g. a bare newline inside a JSON string)

This module provides extract_first_json() as the single entry point used
by all agent _parse_response() methods, replacing ad-hoc strip/loads calls.
"""

from __future__ import annotations

import re


def _fix_control_chars(s: str) -> str:
    """Escape literal control characters inside JSON string values.

    Walks the JSON text character-by-character tracking whether we are
    inside a quoted string, then replaces bare \\n / \\r / other control
    chars with their JSON-escaped counterparts.  Characters outside strings
    are left untouched so structural whitespace is preserved.
    """
    result: list[str] = []
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ord(ch) < 0x20:
            # Other bare control chars inside a string — drop silently.
            pass
        else:
            result.append(ch)

    return "".join(result)


def extract_first_json(content: str) -> str:
    """Extract and sanitize the first JSON object from an LLM response.

    Steps:
    1. Strip markdown code fences (```json / ```).
    2. Extract the first {...} block — discards any trailing text.
    3. Fix unescaped control characters inside string values.

    Returns a clean JSON string ready for json.loads().
    Raises ValueError if no JSON object is found in *content*.
    """
    # Remove markdown code fences that some models wrap around JSON.
    text = re.sub(r"```(?:json)?", "", content).strip()

    # Greedy match of the outermost {...} block.
    # [\s\S]* handles multi-line content inside nested objects.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Nenhum objeto JSON encontrado na resposta do LLM")

    raw = match.group()
    return _fix_control_chars(raw)
