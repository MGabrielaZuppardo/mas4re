"""
Testes de integração do ClassificationAgent contra PROMISE NFR+.
Requer Ollama rodando localmente com qwen2.5:7b.
"""

import pytest
from sklearn.metrics import accuracy_score, classification_report, f1_score

from agents.classifier import ClassificationAgent
from config.settings import settings


@pytest.fixture(scope="module")
def agent():
    return ClassificationAgent(model=settings.classifier_model)


@pytest.fixture(scope="module")
def classified_sample(agent, promise_sample):
    return agent.classify_batch(promise_sample)


class TestClassificadorPromise:
    def test_todos_classificados(self, classified_sample, promise_sample):
        assert len(classified_sample) == len(promise_sample)

    def test_f1_macro_minimo(self, classified_sample, promise_sample):
        """F1 macro mínimo: 0.60 para Ollama local."""
        y_true = [r.metadata["label_type"] for r in promise_sample]
        y_pred = [r.requirement_type.value for r in classified_sample]
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"\nF1 Macro: {f1:.4f}")
        assert f1 >= 0.60

    def test_acuracia_geral(self, classified_sample, promise_sample):
        y_true = [r.metadata["label_type"] for r in promise_sample]
        y_pred = [r.requirement_type.value for r in classified_sample]
        acc = accuracy_score(y_true, y_pred)
        print(f"\nAcurácia: {acc:.4f}")
        assert acc >= 0.60

    def test_confianca_preenchida(self, classified_sample):
        for req in classified_sample:
            assert 0.0 <= req.confidence <= 1.0

    def test_report_por_categoria(self, classified_sample, promise_sample):
        y_true = [r.metadata["label_type"] for r in promise_sample]
        y_pred = [r.requirement_type.value for r in classified_sample]
        print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")


class TestClassificadorIdioma:
    def test_divergencia_en_vs_pt(self, agent, promise_sample):
        """Mede impacto do idioma na classificação."""
        with_en = [r for r in promise_sample if r.text_en]
        if len(with_en) < 3:
            pytest.skip("Poucos requisitos com texto EN.")

        from domain.models import Requirement

        reqs_en = [
            Requirement(id=r.id, text=r.text_en, source="PROMISE_EN", metadata=r.metadata)
            for r in with_en
        ]

        classified_pt = agent.classify_batch(with_en)
        classified_en = agent.classify_batch(reqs_en)

        divergencias = [
            (pt, en)
            for pt, en in zip(classified_pt, classified_en)
            if pt.requirement_type != en.requirement_type
        ]

        taxa = len(divergencias) / len(with_en)
        print(f"\nDivergência EN vs PT: {taxa:.2%}")
