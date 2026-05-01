"""
Converte o dataset PROMISE NFR (.arff) para CSV e traduz para PT-BR.

Uso:
    python scripts/convert_promise.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator

ARFF_PATH = Path("datasets/data/promise_nfr/nfr/nfr.arff")
OUT_EN = Path("datasets/data/promise_nfr/promise_nfr_en.csv")
OUT_PT = Path("datasets/data/promise_nfr/promise_nfr_pt.csv")

# ── Mapeamento de categorias ───────────────────────────────
CATEGORY_MAP = {
    "F": "F",  # Functional
    "A": "A",  # Availability
    "L": "LF",  # Legal → agrupado em Look & Feel
    "LF": "LF",  # Look and Feel
    "MN": "MN",  # Maintainability
    "O": "O",  # Operational
    "PE": "PE",  # Performance
    "SC": "SC",  # Scalability
    "SE": "SE",  # Security
    "US": "US",  # Usability
    "FT": "FT",  # Fault Tolerance
    "PO": "PO",  # Portability
}


def load_arff() -> pd.DataFrame:
    print(f"📂 Carregando: {ARFF_PATH}")
    rows: list[dict] = []
    with open(ARFF_PATH, encoding="utf-8", errors="replace") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.upper() == "@DATA":
                in_data = True
                continue
            if not in_data:
                continue

            parts = line.split(",", 1)
            project_id = parts[0].strip()
            rest = parts[1].strip()
            last_comma = rest.rfind(",")
            req_text = rest[:last_comma].strip().strip("'\"")
            label = rest[last_comma + 1 :].strip()
            rows.append({"ProjectID": int(project_id), "RequirementText": req_text, "class": label})

    df = pd.DataFrame(rows)
    df["class"] = df["class"].map(CATEGORY_MAP).fillna(df["class"])
    print(f"✅ Carregado: {len(df)} requisitos")
    print(df["class"].value_counts().to_string())
    return df


def translate_to_pt(df: pd.DataFrame) -> pd.DataFrame:
    translator = GoogleTranslator(source="en", target="pt")
    translated: list[str] = []

    print(f"\n🌐 Traduzindo {len(df)} requisitos para PT-BR...")

    for i, text in enumerate(df["RequirementText"], start=1):
        try:
            result = translator.translate(text)
            translated.append(result)
        except Exception as e:
            print(f"  ⚠️  Erro na linha {i}: {e} — mantendo original")
            translated.append(text)

        if i % 50 == 0:
            print(f"  → {i}/{len(df)} traduzidos...")

        time.sleep(0.3)

    df["RequirementText_PT"] = translated
    print("✅ Tradução concluída!")
    return df


def save(df: pd.DataFrame) -> None:
    # CSV em inglês
    df[["ProjectID", "RequirementText", "class"]].to_csv(OUT_EN, index=False)
    print(f"💾 Salvo: {OUT_EN}")

    # CSV em português
    df[["ProjectID", "RequirementText", "RequirementText_PT", "class"]].to_csv(OUT_PT, index=False)
    print(f"💾 Salvo: {OUT_PT}")


def main() -> None:
    df = load_arff()
    df = translate_to_pt(df)
    save(df)
    print("\n🎉 Conversão finalizada!")
    print(f"   EN → {OUT_EN}")
    print(f"   PT → {OUT_PT}")


if __name__ == "__main__":
    main()
