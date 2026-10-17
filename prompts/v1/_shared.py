from __future__ import annotations

from domain.enums import Lang
from prompts.localization import localized

# ── Categorias PROMISE NFR+ padrão (usadas quando nfr_categories=None) ────────

_PROMISE_CATEGORIES_PT: list[tuple[str, str]] = [
    ("A", "Availability — disponibilidade do sistema"),
    ("FT", "Fault Tolerance — tolerância a falhas"),
    ("LF", "Look and Feel — aparência e estética"),
    ("MN", "Maintainability — manutenibilidade"),
    ("O", "Operational — requisitos operacionais"),
    ("PE", "Performance — desempenho"),
    ("PO", "Portability — portabilidade"),
    ("SC", "Scalability — escalabilidade"),
    ("SE", "Security — segurança"),
    ("US", "Usability — usabilidade"),
]

_PROMISE_CATEGORIES_EN: list[tuple[str, str]] = [
    ("A", "Availability"),
    ("FT", "Fault Tolerance"),
    ("LF", "Look and Feel"),
    ("MN", "Maintainability"),
    ("O", "Operational"),
    ("PE", "Performance"),
    ("PO", "Portability"),
    ("SC", "Scalability"),
    ("SE", "Security"),
    ("US", "Usability"),
]

_PROMISE_CATEGORIES_ES: list[tuple[str, str]] = [
    ("A", "Availability — disponibilidad del sistema"),
    ("FT", "Fault Tolerance — tolerancia a fallos"),
    ("LF", "Look and Feel — apariencia y estética"),
    ("MN", "Maintainability — mantenibilidad"),
    ("O", "Operational — requisitos operativos"),
    ("PE", "Performance — rendimiento"),
    ("PO", "Portability — portabilidad"),
    ("SC", "Scalability — escalabilidad"),
    ("SE", "Security — seguridad"),
    ("US", "Usability — usabilidad"),
]

_PROMISE_CATEGORIES: dict[Lang, list[tuple[str, str]]] = {
    Lang.PT: _PROMISE_CATEGORIES_PT,
    Lang.EN: _PROMISE_CATEGORIES_EN,
    Lang.ES: _PROMISE_CATEGORIES_ES,
}

# ── Blocos de categorias NFR ───────────────────────────────────────────────────

_NFR_BLOCK_PT = """\
Se NF, atribua EXATAMENTE um dos códigos abaixo para "nfr_category" \
(use apenas o código, ex: "PE", não "desempenho"):
{categories}"""

_NFR_BLOCK_EN = """\
If NF, assign EXACTLY one of the codes below for "nfr_category" \
(use only the code, e.g. "PE", not "performance"):
{categories}"""

_NFR_BLOCK_ES = """\
Si es NF, asigna EXACTAMENTE uno de los códigos siguientes a "nfr_category" \
(usa solo el código, p. ej. "PE", no "rendimiento"):
{categories}"""

_NFR_BLOCK: dict[Lang, str] = {
    Lang.PT: _NFR_BLOCK_PT,
    Lang.EN: _NFR_BLOCK_EN,
    Lang.ES: _NFR_BLOCK_ES,
}


def default_nfr_categories(lang: Lang) -> list[tuple[str, str]]:
    """The PROMISE NFR+ default taxonomy used when an agent has no
    dataset-provided nfr_categories -- same fallback build_nfr_block uses,
    exposed so callers that need the raw (code, description) pairs (e.g.
    agents/tools.py's taxonomy-lookup tool) don't duplicate it."""
    return localized(_PROMISE_CATEGORIES, lang)


def build_nfr_block(
    nfr_categories: list[tuple[str, str]] | None,
    lang: Lang,
) -> str:
    """Renderiza o bloco de categorias NFR no prompt.

    Quando nfr_categories=None usa as categorias PROMISE NFR+ padrão,
    garantindo que o modelo receba sempre os códigos canônicos (A, FT, …)
    em vez de gerar prosa livre.
    """
    cats = nfr_categories if nfr_categories is not None else default_nfr_categories(lang)
    categories_str = "\n".join(f"  - {code}: {desc}" for code, desc in cats)
    return localized(_NFR_BLOCK, lang).format(categories=categories_str)
