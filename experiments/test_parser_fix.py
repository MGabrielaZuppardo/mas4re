"""Smoke test do schema fix no ClassificationAgent._parse_response."""

import logging
import sys

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

from agents.classifier import _NFR_ONLY_CODES, ClassificationAgent  # noqa: E402

# Verificar conjuntos
print("_NFR_ONLY_CODES:", sorted(_NFR_ONLY_CODES))
print(f"  'F'  in NFR_ONLY: {'F' in _NFR_ONLY_CODES}  (deve ser False)")
print(f"  'NF' in NFR_ONLY: {'NF' in _NFR_ONLY_CODES}  (deve ser False)")
print(f"  'PE' in NFR_ONLY: {'PE' in _NFR_ONLY_CODES}  (deve ser True)")
print(f"  'SE' in NFR_ONLY: {'SE' in _NFR_ONLY_CODES}  (deve ser True)")
print()

agent = ClassificationAgent.__new__(ClassificationAgent)
agent.model = "test"

test_cases = [
    # Casos que antes crashavam (mistral coloca categoria no campo errado)
    (
        "PE-wrong-field",
        '{"requirement_type": "PE", "confidence": 0.85, "justification": "performance req"}',
    ),
    (
        "SE-wrong-field",
        '{"requirement_type": "SE", "nfr_category": "SE", "confidence": 0.80, "justification": "security req"}',  # noqa: E501
    ),
    (
        "A-wrong-field",
        '{"requirement_type": "A", "confidence": 0.90, "justification": "availability req"}',
    ),
    # Casos normais que devem permanecer inalterados
    (
        "NF-normal",
        '{"requirement_type": "NF", "nfr_category": "US", "confidence": 0.75, "justification": "usability"}',  # noqa: E501
    ),
    (
        "F-normal",
        '{"requirement_type": "F", "confidence": 0.90, "justification": "functional req"}',
    ),
    # Fallback: JSON inválido
    ("invalid-json", "not json at all"),
]

print(f"  {'Case':<20} {'type':<5} {'nfr':<6} {'conf':<6} {'justification[:45]'}")
print("-" * 80)
for name, tc in test_cases:
    result = agent._parse_response(tc, "test-id")
    nfr_str = result.nfr_category or "None"
    print(
        f"  {name:<20} {result.requirement_type.value:<5} {nfr_str:<6} {result.confidence:.2f}   {result.justification[:45]}"  # noqa: E501
    )
