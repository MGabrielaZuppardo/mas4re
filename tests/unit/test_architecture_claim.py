"""The claim "this run is a multi-agent architecture" is decided from the data, by frozen limits."""

from __future__ import annotations

from pathlib import Path

from experiments.analyze_agency import AgencyRun
from experiments.architecture_claim import (
    INDISTINGUISHABLE,
    MINIMAL,
    NOT_SUPPORTED,
    SUPPORTED,
    evaluate_arm,
    evaluate_arms,
)
from experiments.confirmatory import load_spec

LIMITS = load_spec()["architecture_claim"]


def _activity(entropy=1.2, exhausted=0.0):
    block = {"action_sequence_entropy_bits": entropy, "budget_exhaustion_rate": exhausted}
    return {"classifier": block, "prioritizer": block}


def _run(arm, requests=3, violations=0, activity=None, architecture=True, conformance=True):
    metrics = {
        "agent_activity": _activity() if activity is None else activity,
        "interaction": {"requests": requests, "agree_rate": 0.5, "refuse_rate": 0.0},
    }
    if conformance:
        metrics["conformance"] = {"violations": violations}
    manifest = {"architecture": {"protocol": arm.lower()}} if architecture else {}
    return AgencyRun(Path(arm), manifest, {"metrics": metrics}, {}, [])


def _all(**overrides):
    return {
        "P0": _run("P0", requests=0),
        "P2": _run("P2", **overrides.get("P2", {})),
        "P3": _run("P3", **overrides.get("P3", {})),
    }


class TestArm:
    def test_p2_com_tudo_em_ordem_e_sustentado(self):
        result = evaluate_arm("P2", _run("P2"), LIMITS)

        assert result["verdict"] == SUPPORTED and result["failed"] == []

    def test_p0_em_ordem_e_minimo_pela_dependencia_de_mao_unica(self):
        assert evaluate_arm("P0", _run("P0", requests=0), LIMITS)["verdict"] == MINIMAL

    def test_p2_sem_pedidos_nao_e_sustentado(self):
        result = evaluate_arm("P2", _run("P2", requests=0), LIMITS)

        assert result["verdict"] == NOT_SUPPORTED and result["failed"] == ["channel_used"]

    def test_violacao_de_conformidade_derruba_o_veredito(self):
        result = evaluate_arm("P3", _run("P3", violations=1), LIMITS)

        assert result["failed"] == ["conformance"]

    def test_sem_metrica_de_conformidade_nao_passa(self):
        assert "conformance" in evaluate_arm("P2", _run("P2", conformance=False), LIMITS)["failed"]

    def test_sem_arquitetura_no_manifest_nao_passa(self):
        assert (
            "architecture_recorded"
            in evaluate_arm("P2", _run("P2", architecture=False), LIMITS)["failed"]
        )

    def test_agente_sem_variacao_de_acoes_nao_age_por_conta_propria(self):
        result = evaluate_arm("P2", _run("P2", activity=_activity(entropy=0.0)), LIMITS)

        assert "classifier_acts" in result["failed"] and "prioritizer_acts" in result["failed"]

    def test_estouro_de_orcamento_acima_do_limite_derruba(self):
        result = evaluate_arm("P2", _run("P2", activity=_activity(exhausted=0.5)), LIMITS)

        assert "classifier_acts" in result["failed"]

    def test_recusa_zero_nao_barra_o_veredito(self):
        result = evaluate_arm("P2", _run("P2"), LIMITS)

        assert result["verdict"] == SUPPORTED and result["reported"]["refuse_rate"] == 0.0


class TestOverall:
    def test_todos_os_bracos_em_ordem(self):
        assert evaluate_arms(_all(), LIMITS)["overall"] == SUPPORTED

    def test_p2_e_p3_sem_pedidos_viram_protocolos_indistinguiveis(self):
        result = evaluate_arms(_all(P2={"requests": 0}, P3={"requests": 0}), LIMITS)

        assert result["overall"] == NOT_SUPPORTED and result["note"] == INDISTINGUISHABLE

    def test_so_um_protocolo_sem_pedidos_nao_gera_a_nota(self):
        result = evaluate_arms(_all(P3={"requests": 0}), LIMITS)

        assert result["overall"] == NOT_SUPPORTED and result["note"] is None

    def test_limites_vem_do_spec_congelado(self):
        assert LIMITS["max_conformance_violations"] == 0
