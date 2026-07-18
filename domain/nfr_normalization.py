from __future__ import annotations

import logging

from domain.enums import NFRCategory, RequirementType

logger = logging.getLogger(__name__)

_VALID_CODES: frozenset[str] = frozenset(c.value for c in NFRCategory)
NFR_ONLY_CODES: frozenset[str] = _VALID_CODES - frozenset(t.value for t in RequirementType)

_PROSE_TO_CODE: dict[str, str] = {
    # Português
    "disponibilidade": "A",
    "tolerância a falhas": "FT",
    "tolerancia a falhas": "FT",
    "tolerância": "FT",
    "aparência": "LF",
    "aparencia": "LF",
    "estética": "LF",
    "estetica": "LF",
    "look and feel": "LF",
    "manutenibilidade": "MN",
    "manutenção": "MN",
    "manutencao": "MN",
    "flexibilidade": "MN",
    "documentação": "MN",
    "suporte": "MN",
    "operacional": "O",
    "operacionalidade": "O",
    "tempo de market": "O",
    "formato de dados": "O",
    "desempenho": "PE",
    "tempo de resposta": "PE",
    "portabilidade": "PO",
    "compatibilidade": "PO",
    "globalização": "PO",
    "internacionalização": "PO",
    "escalabilidade": "SC",
    "segurança": "SE",
    "seguranca": "SE",
    "usabilidade": "US",
    "acessibilidade": "US",
    # English
    "availability": "A",
    "fault tolerance": "FT",
    "reliability": "FT",
    "reliabilidade": "FT",
    "confiabilidade": "FT",
    "maintainability": "MN",
    "flexibility": "MN",
    "documentation": "MN",
    "operational": "O",
    "time to market": "O",
    "data format": "O",
    "performance": "PE",
    "response time": "PE",
    "portability": "PO",
    "compatibility": "PO",
    "internationalization": "PO",
    "globalization": "PO",
    "scalability": "SC",
    "security": "SE",
    "usability": "US",
    "accessibility": "US",
}


def normalize_nfr_category(raw: str | None) -> str | None:
    """Normalize a raw nfr_category value returned by an LLM to a PROMISE code.

    Accepts an already-valid code (case-insensitive) or common PT/EN prose
    variants (e.g. "desempenho" -> "PE"). Unrecognized values are discarded
    (logged, returns None) instead of leaking free-form text into metrics.
    """
    if raw is None:
        return None
    upper = raw.strip().upper()
    if upper in _VALID_CODES:
        return upper
    lower = raw.strip().lower()
    if lower in _PROSE_TO_CODE:
        code = _PROSE_TO_CODE[lower]
        logger.warning("nfr_category prosa mapeada | raw=%r -> %s", raw, code)
        return code
    logger.warning("nfr_category nao reconhecida, descartada | raw=%r", raw)
    return None
