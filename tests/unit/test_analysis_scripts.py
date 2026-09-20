from unittest.mock import MagicMock, patch

import numpy as np

from domain.models import ClassificationOutput, PrioritizationOutput
from experiments.analyze_coordination_power import conflict_ceiling, summarize
from experiments.analyze_error_signals import (
    disagreement_report,
    majority_vote_accuracy,
    signal_report,
)
from experiments.analyze_verifier_oracle import out_of_fold_nfr_probability, verifier_report
from experiments.diagnose_retry_determinism import differences


def _pred(text="t", requirement_type="F", label="F", category=None, confidence=0.9, priority="M"):
    return {
        "text": text,
        "requirement_type": requirement_type,
        "nfr_category": category,
        "confidence": confidence,
        "priority": priority,
        "justification": "ok",
        "metadata": {"label_type": label},
    }


class TestSignalReport:
    def test_taxas_e_captura(self):
        items = [
            _pred(requirement_type="NF", label="F", confidence=0.5),
            _pred(requirement_type="F", label="F", confidence=0.9),
            _pred(requirement_type="F", label="NF", confidence=0.9),
            _pred(requirement_type="F", label="F", confidence=0.95),
        ]

        report = signal_report(items, lambda p: p["confidence"] < 0.70)

        assert report["flagged"] == 1
        assert report["error_rate_flagged"] == 1.0
        assert report["error_rate_unflagged"] == 1 / 3
        assert report["base_error_rate"] == 0.5
        assert report["errors_captured"] == 0.5

    def test_sinal_que_nao_marca_nada_devolve_none(self):
        report = signal_report([_pred(), _pred()], lambda p: False)

        assert report["flagged"] == 0
        assert report["error_rate_flagged"] is None
        assert report["errors_captured"] is None


class TestModelDisagreement:
    @staticmethod
    def _per_model():
        return {
            "a": {"t1": _pred("t1", "F", "F"), "t2": _pred("t2", "F", "NF")},
            "b": {"t1": _pred("t1", "F", "F"), "t2": _pred("t2", "NF", "NF")},
        }

    def test_relatorio_de_discordancia(self):
        report = disagreement_report(self._per_model(), "a", ["t1", "t2"])

        assert report["flagged"] == 1
        assert report["error_rate_disagree"] == 1.0
        assert report["error_rate_unanimous"] == 0.0
        assert report["errors_captured"] == 1.0

    def test_empate_no_voto_resolve_para_o_primeiro_modelo_ordenado(self):
        accuracy = majority_vote_accuracy(self._per_model(), ["t1", "t2"])

        # t2 ties 1-1; "a" (first) votes F but the label is NF, so only t1 is right
        assert accuracy == 0.5


class TestCoordinationPower:
    def test_teto_do_efeito_a_partir_dos_conflitos(self):
        preds = [
            _pred(category="SE", priority="C"),
            _pred(category="SE", priority="M"),
            _pred(category="US", priority="C"),
            _pred(),
        ]

        ceiling = conflict_ceiling(preds)

        assert ceiling == {"conflicts": 1, "critical_items": 2, "effect_ceiling": 0.25}

    def test_resumo_conta_execucoes_acima_dos_limites(self):
        rows = [
            {"strategy": "pipeline", "must_share": 0.9, "conflicts": 0},
            {"strategy": "pipeline", "must_share": 0.5, "conflicts": 3},
            {"strategy": "baseline", "must_share": 0.7},
            {"strategy": "baseline", "must_share": None},
        ]

        summary = summarize(rows, must_limit=0.60, strict_limit=0.80)

        assert summary["pipeline_runs"] == 2
        assert summary["pipeline_runs_with_conflict"] == 1
        assert summary["total_conflicts"] == 3
        assert summary["runs_over_0.60"] == 2
        assert summary["runs_over_0.80"] == 1


class TestVerifier:
    def test_ganho_quando_o_verificador_confiante_sobrescreve(self):
        truth = np.array([1, 1, 0, 0], dtype=bool)
        llm = np.array([1, 0, 0, 1], dtype=bool)
        verifier = np.array([1, 1, 0, 0], dtype=bool)
        confidence = np.array([0.9, 0.9, 0.9, 0.6])

        report = verifier_report(llm, truth, verifier, confidence, min_confidence=0.85)

        assert report["confident_disagreements"] == 1
        assert report["llm_error_rate_when_confident"] == 1.0
        assert report["llm_errors_captured"] == 1.0
        assert report["accuracy_gain_if_verifier_overrides"] == 0.25

    def test_probabilidades_fora_da_dobra_sao_deterministicas(self):
        texts = [f"{'security' if i % 2 else 'display'} item {i % 3}" for i in range(12)]
        is_nfr = np.array([i % 2 == 1 for i in range(12)])
        groups = np.array([i % 4 for i in range(12)])

        first = out_of_fold_nfr_probability(texts, is_nfr, groups, n_splits=4)
        second = out_of_fold_nfr_probability(texts, is_nfr, groups, n_splits=4)

        assert first.shape == (12,)
        assert np.array_equal(first, second)


class TestRetryDeterminism:
    def test_lista_apenas_itens_que_mudaram(self):
        first = {"a": ("F", None, 0.9), "b": ("NF", "SE", 0.9)}
        second = {"a": ("F", None, 0.9), "b": ("F", None, 0.8)}

        assert differences(first, second) == [
            {"id": "b", "first": ["NF", "SE", 0.9], "second": ["F", None, 0.8]}
        ]


class TestClassifierVariants:
    @staticmethod
    def _output(**overrides) -> ClassificationOutput:
        base = {
            "requirement_id": "r1",
            "requirement_type": "F",
            "confidence": 0.9,
            "justification": "ok",
        }
        return ClassificationOutput(**{**base, **overrides})

    def test_always_critique_dispara_mesmo_com_saida_confiavel(self):
        from experiments.diagnose_classifier_variants import AlwaysCritiqueClassifier

        with patch("agents.classifier.build_llm"):
            agent = AlwaysCritiqueClassifier(model="ollama/m")

        assert agent._needs_critique(self._output()) is True

    def test_no_critique_nao_dispara_mesmo_com_baixa_confianca(self):
        from experiments.diagnose_classifier_variants import NoCritiqueClassifier

        with patch("agents.classifier.build_llm"):
            agent = NoCritiqueClassifier(model="ollama/m")

        assert agent._needs_critique(self._output(confidence=0.1)) is False

    def test_contador_registra_a_virada_de_tipo(self):
        from experiments.diagnose_classifier_variants import CurrentClassifier

        with patch("agents.classifier.build_llm"):
            agent = CurrentClassifier(model="ollama/m")
        revision = MagicMock()
        revision.content = (
            '{"needs_revision": true, "revised_output": {"requirement_type": "NF", '
            '"nfr_category": "SE", "confidence": 0.9, "justification": "x"}}'
        )
        agent._llm.invoke = MagicMock(return_value=revision)

        agent._critique("texto", self._output())

        assert agent.stats["critique_calls"] == 1
        assert agent.stats["flip_F_to_NF"] == 1

    def test_sem_memoria_deixa_a_memoria_desligada(self):
        from experiments.diagnose_classifier_variants import NoMemoryClassifier

        with patch("agents.classifier.build_llm"):
            agent = NoMemoryClassifier(model="ollama/m")

        assert agent.classify_batch([]) == []
        assert agent._memory is None


class TestPrioritizerVariants:
    @staticmethod
    def _output(priority="M") -> PrioritizationOutput:
        return PrioritizationOutput(
            requirement_id="r1",
            priority=priority,
            priority_score=1.0,
            priority_rank=1,
            justification="ok",
        )

    def test_contador_registra_a_troca_de_prioridade(self):
        from experiments.diagnose_prioritizer_variants import CurrentPrioritizer

        with patch("agents.prioritizer.build_llm"):
            agent = CurrentPrioritizer(model="ollama/m")
        revision = MagicMock()
        revision.content = (
            '{"needs_revision": true, "revised_output": {"priority": "S", '
            '"priority_score": 0.75, "priority_rank": 1, "justification": "x"}}'
        )
        agent._llm.invoke = MagicMock(return_value=revision)

        agent._critique("texto", self._output())

        assert agent.stats["critique_calls"] == 1
        assert agent.stats["flip_M_to_S"] == 1

    def test_sem_critica_nao_chama_o_llm(self):
        from experiments.diagnose_prioritizer_variants import NoCritiquePrioritizer

        with patch("agents.prioritizer.build_llm"):
            agent = NoCritiquePrioritizer(model="ollama/m")
        original = self._output()

        assert agent._critique("texto", original) is original
        agent._llm.invoke.assert_not_called()
