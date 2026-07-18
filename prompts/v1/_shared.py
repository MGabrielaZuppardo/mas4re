from __future__ import annotations

from domain.enums import Lang

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

# ── Blocos de categorias NFR ───────────────────────────────────────────────────

_NFR_BLOCK_PT = """\
Se NF, atribua EXATAMENTE um dos códigos abaixo para "nfr_category" \
(use apenas o código, ex: "PE", não "desempenho"):
{categories}"""

_NFR_BLOCK_EN = """\
If NF, assign EXACTLY one of the codes below for "nfr_category" \
(use only the code, e.g. "PE", not "performance"):
{categories}"""


def build_nfr_block(
    nfr_categories: list[tuple[str, str]] | None,
    lang: Lang,
) -> str:
    """Renderiza o bloco de categorias NFR no prompt.

    Quando nfr_categories=None usa as categorias PROMISE NFR+ padrão,
    garantindo que o modelo receba sempre os códigos canônicos (A, FT, …)
    em vez de gerar prosa livre.
    """
    cats = (
        nfr_categories
        if nfr_categories is not None
        else (_PROMISE_CATEGORIES_PT if lang is Lang.PT else _PROMISE_CATEGORIES_EN)
    )
    categories_str = "\n".join(f"  - {code}: {desc}" for code, desc in cats)
    template = _NFR_BLOCK_PT if lang is Lang.PT else _NFR_BLOCK_EN
    return template.format(categories=categories_str)
