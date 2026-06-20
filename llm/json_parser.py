"""Robust JSON extraction for LLM responses.

Small models (phi3.5, llama3.1, etc.) produce several failure modes:
  - Extra text / second JSON block after closing brace -> "Extra data"
  - Unescaped control characters (bare \\n inside strings) -> "Invalid control character"
  - Unescaped double-quotes inside string values -> "Expecting ',' delimiter"
  - Double-close pattern "word"" -> same error

This module provides extract_first_json() as the single entry point used
by all agent _parse_response() methods, and coerce_str() for fields that
small models sometimes return as nested dicts instead of plain strings.
"""

from __future__ import annotations

import re


def _next_nonws(s: str, start: int) -> tuple[int, str]:
    """Return (index, char) of the first non-whitespace char at or after *start*.

    Returns (len(s), '') if the end of the string is reached.
    """
    i = start
    n = len(s)
    while i < n and s[i] in " \t\r\n":
        i += 1
    return (i, s[i] if i < n else "")


def _sanitize_json_string(s: str) -> str:  # noqa: C901
    """Fix common LLM JSON issues via a single-pass state machine.

    Context-aware approach:
    - KEY strings  (after { or ,): always close at the first unescaped ".
    - VALUE strings (after :): use look-ahead to distinguish a real
      end-of-string from an embedded unescaped quote.

    Look-ahead rule for VALUE strings — when we see a candidate closing ":
      * next non-ws in {, } ] ''}:  real end-of-string.
      * next non-ws is another '"':  two-level look-ahead:
          - char after THAT '"' in {, } ] ''}:  the first '"' is embedded,
            the second '"' will close the string (double-close pattern).
          - otherwise:  first '"' is a real end-of-string (next '"' is a key).
      * anything else:  embedded quote — escape with backslash.

    Additional fixes inside strings:
      - Bare \\n / \\r  ->  \\\\n / \\\\r
      - Other control chars (ord < 0x20)  ->  dropped
    """
    CLOSERS = frozenset((",", "}", "]", '"', ""))

    result: list[str] = []
    in_string = False
    in_key = False  # True while inside a key string
    escape_next = False
    expecting_key = True  # True right after { or ,
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        # ── already-escaped sequence ─────────────────────────────────────
        if escape_next:
            result.append(ch)
            escape_next = False
            i += 1
            continue

        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            i += 1
            continue

        # ── quote character ───────────────────────────────────────────────
        if ch == '"':
            if not in_string:
                in_string = True
                in_key = expecting_key
                result.append(ch)

            elif in_key:
                # Key strings always close at the first unescaped quote.
                in_string = False
                in_key = False
                expecting_key = False  # ':' will follow
                result.append(ch)

            else:
                # Value string — look ahead.
                j, next_ch = _next_nonws(s, i + 1)

                if next_ch == '"':
                    # Two-level look-ahead: check what comes after the next ".
                    _, after_next = _next_nonws(s, j + 1)
                    if after_next in (",", "}", "]", ""):
                        # Double-close pattern "word"" } — embed current ".
                        result.append('\\"')
                    else:
                        # Next " opens the next key — current " ends string.
                        in_string = False
                        if next_ch == ",":
                            expecting_key = True
                        result.append(ch)

                elif next_ch in CLOSERS:
                    # Real end-of-string.
                    in_string = False
                    if next_ch == ",":
                        expecting_key = True
                    result.append(ch)

                else:
                    # Embedded quote — escape it.
                    result.append('\\"')

            i += 1
            continue

        # ── outside a string ─────────────────────────────────────────────
        if not in_string:
            if ch == ":":
                expecting_key = False
            elif ch in (",", "{"):
                expecting_key = True
            result.append(ch)

        # ── inside a string ──────────────────────────────────────────────
        else:
            if ch == "\n":
                result.append("\\n")
            elif ch == "\r":
                result.append("\\r")
            elif ord(ch) < 0x20:
                pass  # drop other bare control chars
            else:
                result.append(ch)

        i += 1

    return "".join(result)


def _extract_balanced(text: str) -> str:
    """Return the first balanced {...} block in *text*.

    Uses a brace-depth counter with string tracking so that braces inside
    quoted values are not counted.  Stops as soon as the opening brace's
    matching closing brace is found — prevents the "Extra data" problem
    when the model appends a second JSON block after the first one.

    The string tracker applies the same double-close look-ahead used by
    _sanitize_json_string so that the "word"" pattern (where phi3.5 emits
    an extra closing quote) does not prematurely end the tracked string,
    which would cause the } counter to miss the real closing brace.

    Raises ValueError if no complete balanced block is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("Nenhum objeto JSON encontrado na resposta do LLM")

    depth = 0
    in_str = False
    esc = False
    i = start
    n = len(text)

    while i < n:
        ch = text[i]

        if esc:
            esc = False
            i += 1
            continue

        if ch == "\\" and in_str:
            esc = True
            i += 1
            continue

        if ch == '"':
            if not in_str:
                in_str = True
            else:
                # Check for double-close pattern: "" followed by , } ] or end.
                # In that case, stay in the string (treat current " as embedded).
                j, next_ch = _next_nonws(text, i + 1)
                if next_ch == '"':
                    _, after_next = _next_nonws(text, j + 1)
                    if after_next in (",", "}", "]", ""):
                        # Double-close — skip this quote, stay in string.
                        i += 1
                        continue
                # Normal close.
                in_str = False
            i += 1
            continue

        if in_str:
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

        i += 1

    raise ValueError("JSON incompleto: bloco {...} sem fechamento encontrado")


def extract_first_json(content: str) -> str:
    """Extract and sanitize the first JSON object from an LLM response.

    Steps:
    1. Strip markdown code fences (```json / ```).
    2. Extract the first *balanced* {...} block (stops at the matching },
       not the last } in the text — fixes "Extra data" from second blocks).
    3. Sanitize: fix control chars and embedded/double-close quotes.

    Returns a clean JSON string ready for json.loads().
    Raises ValueError if no JSON object is found in *content*.
    """
    text = re.sub(r"```(?:json)?", "", content).strip()
    raw = _extract_balanced(text)
    return _sanitize_json_string(raw)


def coerce_str(value: object, sep: str = "; ") -> str:
    """Coerce *value* to a plain string.

    Small models (phi3.5) sometimes emit a nested dict where a plain string
    is expected, e.g.:
        "justification": {"impacto no negocio": "...", "urgencia": "..."}

    This helper flattens any dict to "key: value; key: value", leaves
    strings unchanged, and converts everything else via str().
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return sep.join(f"{k}: {v}" for k, v in value.items())
    return str(value)
