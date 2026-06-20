import math


def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def h_magnitude(h):
    ah = abs(h)
    if ah < 0.20:
        return "small"
    if ah < 0.50:
        return "medium"
    return "large"


rq1 = [
    ("qwen2.5:7b", "PT", 0.7335, 0.8250, +0.091),
    ("qwen2.5:7b", "EN", 0.6806, 0.7994, +0.119),
    ("llama3.1:8b", "PT", 0.6356, 0.6966, +0.061),
    ("llama3.1:8b", "EN", 0.6726, 0.7287, +0.056),
    ("mistral:7b", "PT", 0.5923, 0.5506, -0.042),
    ("mistral:7b", "EN", 0.5875, 0.6196, +0.032),
]

print("RQ1 — Cliff's delta vs Cohen's h")
print(
    f"{'Modelo':<14} {'Lang':<4} {'base':>6} {'pipe':>6} {'Dacc':>7}  {'Cliff-d':>12}  {'Cohen-h':>14}  mag-h"  # noqa: E501
)
print("-" * 80)

for model, lang, base, pipe, cd in rq1:
    h = cohens_h(pipe, base)
    mag = h_magnitude(h)
    dacc = pipe - base
    cd_mag = "negligible"
    print(
        f"{model:<14} {lang:<4} {base:.4f}  {pipe:.4f}  {dacc:+.4f}   {cd:+.3f}({cd_mag})   {h:+.4f}   {mag}"  # noqa: E501
    )

print()
print("Thresholds Cohen h : small>=0.20 | medium>=0.50 | large>=0.80")
print("Thresholds Cliff d : negligible<0.147 | small<0.330 | medium<0.474 | large>=0.474")
