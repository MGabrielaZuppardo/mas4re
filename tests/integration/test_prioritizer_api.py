"""
Testes de integração do PrioritizationAgent contra PROMISE NFR+.
Requer Ollama rodando localmente com llama3.1:8b.
"""
import pytest
from collections import Counter
from scipy.stats import kendalltau

from agents.prioritizer import PrioritizationAgent
from config.settings import settings
from domain.enums import MoSCoWPriority


@pytest.fixture(scope="module")
def agent():
    return PrioritizationAgent(model=settings.prioritizer_model)


@pytest.fixture(scope="module")
def prioritized_sample(agent, promise_sample):
    """Classifica amostra e prioriza — pipeline completo."""
    from agents.classifier import ClassificationAgent
    classifier = ClassificationAgent(model=settings.classifier_model)
    classified = classifier.classify_batch(promise_sample)
    return agent.prioritize_batch(classified)


class TestPriorizadorPromise:

    def test_todos_priorizados(self, prioritized_sample, promise_sample):
        assert len(prioritized_sample) == len(promise_sample)

    def test_priority_preenchida(self, prioritized_sample):
        for req in prioritized_sample:
            assert req.priority in MoSCoWPriority

    def test_priority_score_range(self, prioritized_sample):
        for req in prioritized_sample:
            assert 0.0 <= (req.priority_score or 0.0) <= 1.0

    def test_ranking_unico_e_sequencial(self, prioritized_sample):
        ranks = [r.priority_rank for r in prioritized_sample if r.priority_rank]
        assert len(ranks) == len(prioritized_sample)
        assert sorted(ranks) == list(range(1, len(ranks) + 1))

    def test_justification_nao_vazia(self, prioritized_sample):
        vazias = [r for r in prioritized_sample if not r.justification_priority]
        taxa_vazia = len(vazias) / len(prioritized_sample)
        print(f"\nJustificativas vazias: {taxa_vazia:.2%}")
        assert taxa_vazia <= 0.10  # máx 10% sem justificativa

    def test_distribuicao_moscow(self, prioritized_sample):
        """Verifica que a distribuição MoSCoW não está degenerada (tudo M)."""
        from evaluation.metrics.prioritization import compute_moscow_distribution
        dist = compute_moscow_distribution(prioritized_sample)
        print(f"\nDistribuição MoSCoW: {dist}")
        # Nenhuma categoria deve concentrar mais de 80% dos requisitos
        for cat, prop in dist.items():
            assert prop <= 0.80, f"Categoria {cat} concentra {prop:.2%} — distribuição degenerada"

    def test_kendall_tau_consistencia(self, prioritized_sample):
        """Verifica correlação interna: score alto → rank baixo (1 = mais prioritário)."""
        scores = [r.priority_score or 0.0 for r in prioritized_sample]
        ranks  = [r.priority_rank or 0 for r in prioritized_sample]
        # Rank é crescente, score é decrescente → correlação negativa esperada
        tau, _ = kendalltau(scores, ranks)
        print(f"\nKendall Tau (score vs rank): {tau:.4f}")
        assert tau <= -0.5, f"Correlação fraca entre score e rank: tau={tau:.4f}"


class TestPriorizadorIdioma:

    def test_divergencia_moscow_en_vs_pt(self, agent, promise_sample):
        """Mede impacto do idioma na priorização MoSCoW."""
        from agents.classifier import ClassificationAgent
        from domain.models import Requirement

        with_en = [r for r in promise_sample if r.text_en]
        if len(with_en) < 3:
            pytest.skip("Poucos requisitos com texto EN.")

        classifier = ClassificationAgent(model=settings.classifier_model)

        classified_pt = classifier.classify_batch(with_en)
        reqs_en = [
            Requirement(id=r.id, text=r.text_en, source="PROMISE_EN", metadata=r.metadata)
            for r in with_en
        ]
        classified_en = classifier.classify_batch(reqs_en)

        prioritized_pt = agent.prioritize_batch(classified_pt)
        prioritized_en = agent.prioritize_batch(classified_en)

        pt_map = {r.id: r for r in prioritized_pt}
        divergencias = [
            r for r in prioritized_en
            if r.id in pt_map and r.priority != pt_map[r.id].priority
        ]

        taxa = len(divergencias) / len(with_en)
        print(f"\nDivergência MoSCoW EN vs PT: {taxa:.2%}")
        # Registra mas não falha — é uma métrica observacional