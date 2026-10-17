"""Priorities against the dataset's human MoSCoW labels."""

from __future__ import annotations

import json

import pytest

from domain.enums import MoSCoWPriority, RequirementType
from domain.models import PrioritizedRequirement, Requirement
from evaluation.metrics.prioritization import compute_priority_gold_metrics

M, S, C, W = (
    MoSCoWPriority.MUST_HAVE,
    MoSCoWPriority.SHOULD_HAVE,
    MoSCoWPriority.COULD_HAVE,
    MoSCoWPriority.WONT_HAVE,
)


def _requirements(golds: list[MoSCoWPriority | None], relevance: list[int] | None = None):
    return [
        Requirement(
            id=f"r{i}",
            text=f"Requisito de teste número {i}",
            metadata={
                **({"ground_truth_priority": gold} if gold is not None else {}),
                **({"ground_truth_relevance": relevance[i]} if relevance else {}),
            },
        )
        for i, gold in enumerate(golds)
    ]


def _predictions(priorities: list[MoSCoWPriority], skip: set[int] = frozenset()):
    return [
        PrioritizedRequirement(
            id=f"r{i}",
            text=f"Requisito de teste número {i}",
            requirement_type=RequirementType.FUNCTIONAL,
            confidence=0.9,
            priority=priority,
            priority_score=priority.score,
            priority_rank=i + 1,
        )
        for i, priority in enumerate(priorities)
        if i not in skip
    ]


class TestAgreement:
    def test_previsao_perfeita(self):
        golds = [M, S, C, W, M, S]

        result = compute_priority_gold_metrics(_predictions(golds), _requirements(golds))

        assert result["accuracy"] == 1.0
        assert result["weighted_kappa"] == 1.0
        assert result["spearman"] == 1.0 and result["kendall_tau_b"] == 1.0
        assert result["mae_ordinal"] == 0.0
        assert result["coverage"] == 1.0 and result["n_evaluated"] == 6

    def test_tudo_must_acerta_so_a_parcela_de_must_e_o_kappa_e_zero(self):
        golds = [M, M, S, S, S, C]

        result = compute_priority_gold_metrics(_predictions([M] * 6), _requirements(golds))

        assert result["accuracy"] == pytest.approx(2 / 6, abs=1e-4)
        assert result["majority_baseline_accuracy"] == 0.5
        assert result["weighted_kappa"] == 0.0
        assert result["per_class"]["M"]["recall"] == 1.0
        assert result["per_class"]["S"]["recall"] == 0.0

    def test_erro_ordinal_pesa_pela_distancia(self):
        golds = [M, M]

        near = compute_priority_gold_metrics(_predictions([S, S]), _requirements(golds))
        far = compute_priority_gold_metrics(_predictions([W, W]), _requirements(golds))

        assert near["mae_ordinal"] == 1.0 and far["mae_ordinal"] == 3.0

    def test_matriz_de_confusao_tem_gabarito_nas_linhas_e_previsao_nas_colunas(self):
        golds = [M, M, S]

        result = compute_priority_gold_metrics(_predictions([M, S, S]), _requirements(golds))

        assert result["confusion_matrix"]["M"] == {"W": 0, "C": 0, "S": 1, "M": 1}
        assert result["confusion_matrix"]["S"]["S"] == 1


class TestCoverageAndAbsence:
    def test_item_nao_processado_baixa_a_cobertura_e_fora_da_conta(self):
        golds = [M, S, S, W]

        result = compute_priority_gold_metrics(
            _predictions([M, S, S, W], skip={3}), _requirements(golds)
        )

        assert (result["n_total"], result["n_evaluated"], result["coverage"]) == (4, 3, 0.75)
        assert result["accuracy"] == 1.0

    def test_sem_gabarito_devolve_vazio(self):
        assert compute_priority_gold_metrics(_predictions([M]), _requirements([None])) == {}

    def test_nada_priorizado_devolve_so_a_cobertura(self):
        result = compute_priority_gold_metrics([], _requirements([M, S]))

        assert result == {"n_total": 2, "n_evaluated": 0, "coverage": 0.0}

    def test_gabarito_lido_de_json_como_texto(self):
        requirement = Requirement(
            id="r0", text="Requisito de teste", metadata={"ground_truth_priority": "M"}
        )

        assert requirement.gold_priority is M


class TestUndefinedStatistics:
    def test_entrada_constante_vira_nulo_e_o_resultado_continua_json_valido(self):
        golds = [S, S, S]

        result = compute_priority_gold_metrics(_predictions([S, S, S]), _requirements(golds))

        assert result["spearman"] is None and result["kendall_tau_b"] is None
        json.dumps(result, allow_nan=False)


class TestRelevance:
    def test_correlacao_do_score_com_a_relevancia_quando_ela_existe(self):
        golds = [M, S, C, W]

        result = compute_priority_gold_metrics(
            _predictions(golds), _requirements(golds, relevance=[4, 3, 2, 1])
        )

        assert result["spearman_score_vs_relevance"] == 1.0

    def test_sem_relevancia_a_chave_nao_aparece(self):
        golds = [M, S]

        assert "spearman_score_vs_relevance" not in compute_priority_gold_metrics(
            _predictions(golds), _requirements(golds)
        )
