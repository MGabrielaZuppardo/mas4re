"""
MAS4RE — Taxa de conformidade de nfr_category (compliance estrutural fina)

§6.6 usa a taxa de confidence==0.0 (falha total de parsing) como proxy de
"structured JSON output compliance". Esse proxy não mostra elevação para
mistral:7b (no máximo 2/625 em qualquer condição), o que aparentemente
contradiz a alegação de "weakest structured JSON output compliance" em
§6.1. Este script mede um proxy mais fino: entre as predições classificadas
como NF (JSON válido, confidence definida), qual fração não produz uma
nfr_category válida/reconhecida. Essa falha não aparece no proxy de §6.6
porque o parsing da resposta como um todo ainda é bem-sucedido.

Uso (a partir de D:/mas4re):
    python experiments/compute_nfr_category_compliance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.compute_stats import (
    LANGS,
    MODELS,
    RESULTS_DIR,
    _find_latest_csv,
    load_all_runs,
    load_csv,
    load_two_call_runs,
)

OUTPUT_PATH = RESULTS_DIR / "nfr_category_compliance.json"


def compliance_for(preds: dict) -> dict:
    nf_total = 0
    nf_missing_category = 0
    zero_conf = 0
    for p in preds.values():
        rt = (p.get("requirement_type") or "").upper()
        conf = p.get("confidence")
        if conf == 0.0:
            zero_conf += 1
        if rt == "NF":
            nf_total += 1
            if not p.get("nfr_category"):
                nf_missing_category += 1
    rate = (nf_missing_category / nf_total) if nf_total else 0.0
    return {
        "nf_total": nf_total,
        "nf_missing_category": nf_missing_category,
        "missing_category_rate": round(rate, 4),
        "zero_confidence_fallbacks": zero_conf,
    }


def main() -> None:
    csv_path = _find_latest_csv()
    run_meta = load_csv(csv_path)
    all_runs = load_all_runs(run_meta)
    all_runs.update(load_two_call_runs())

    print(f"\n{'=' * 88}")
    print("Taxa de nfr_category ausente/inválida entre predições NF, por condição")
    print("(proxy de compliance estrutural fina, distinto do zero-confidence de §6.6)")
    print("=" * 88)

    results = []
    for strategy in ["baseline", "pipeline", "two_call_baseline"]:
        for model in MODELS:
            for lang in LANGS:
                preds = all_runs.get((strategy, model, lang), {})
                if not preds:
                    continue
                stats = compliance_for(preds)
                if stats["nf_missing_category"] == 0:
                    continue
                results.append({"strategy": strategy, "model": model, "lang": lang, **stats})
                print(
                    f"  {strategy:<18} {model:<14} {lang.upper()}  "
                    f"NF={stats['nf_total']:>4}  "
                    f"sem_categoria={stats['nf_missing_category']:>3}  "
                    f"taxa={stats['missing_category_rate']:.4f}  "
                    f"zero_conf={stats['zero_confidence_fallbacks']}"
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
