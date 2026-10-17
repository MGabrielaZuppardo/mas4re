"""The mediator of protocol P3: what it finds, whom it asks, and in which language."""

from __future__ import annotations

from domain.enums import Lang, MoSCoWPriority, RequirementType
from domain.messages import Performative, ReprioritizationRequest
from domain.models import PrioritizedRequirement
from pipeline.mediation import (
    MEDIATOR,
    FindingKind,
    find_conflicts,
    find_must_inflation,
    mediate,
    must_share,
    request_for,
)

M, S, C = MoSCoWPriority.MUST_HAVE, MoSCoWPriority.SHOULD_HAVE, MoSCoWPriority.COULD_HAVE


def _item(
    i: int,
    priority: MoSCoWPriority,
    score: float | None = None,
    category: str | None = None,
    rank: int | None = None,
) -> PrioritizedRequirement:
    return PrioritizedRequirement(
        id=f"r{i}",
        text=f"Requisito de teste número {i}",
        requirement_type=(
            RequirementType.NON_FUNCTIONAL if category else RequirementType.FUNCTIONAL
        ),
        nfr_category=category,
        confidence=0.9,
        priority=priority,
        priority_score=priority.score if score is None else score,
        priority_rank=rank or i + 1,
    )


class TestMustShare:
    def test_fracao_de_musts(self):
        assert must_share([_item(0, M), _item(1, S), _item(2, M), _item(3, C)]) == 0.5

    def test_backlog_vazio(self):
        assert must_share([]) == 0.0


class TestConflicts:
    def test_categoria_critica_com_prioridade_baixa_e_conflito(self):
        items = [_item(0, C, category="SE"), _item(1, C, category="US"), _item(2, M, category="SE")]

        [finding] = find_conflicts(items, Lang.PT)

        assert (finding.requirement_id, finding.kind) == ("r0", FindingKind.CONFLICT)
        assert "'SE'" in finding.evidence and "'C'" in finding.evidence


class TestMustInflation:
    def test_no_limite_ou_abaixo_nao_ha_achado(self):
        items = [_item(i, M) for i in range(3)] + [_item(i, S) for i in range(3, 5)]

        assert find_must_inflation(items, Lang.PT) == []

    def test_pede_so_o_excesso_e_os_musts_de_menor_score(self):
        scores = [1.0, 0.9, 0.8, 0.7]
        items = [_item(i, M, score) for i, score in enumerate(scores)]

        findings = find_must_inflation(items, Lang.PT)

        # 4 Musts of 4 (100%); the limit allows floor(0.6 * 4) = 2, so 2 must be reconsidered.
        assert [f.requirement_id for f in findings] == ["r3", "r2"]
        assert all(f.kind is FindingKind.MUST_INFLATION for f in findings)
        assert "100%" in findings[0].evidence and "60%" in findings[0].evidence

    def test_empate_de_score_pede_primeiro_o_de_pior_posicao(self):
        items = [_item(0, M, 0.9, rank=1), _item(1, M, 0.9, rank=2), _item(2, S)]

        [finding] = find_must_inflation(items, Lang.PT, limit=0.5)

        assert finding.requirement_id == "r1"


class TestMediate:
    def test_junta_conflitos_e_inflacao_com_conflitos_primeiro(self):
        # 5 Musts of 6 (83%); the limit allows floor(0.6 * 6) = 3, so the 2 weakest are asked.
        items = [_item(0, C, category="SE")] + [_item(i, M, 1.0 - i / 10) for i in range(1, 6)]

        report = mediate(items, Lang.PT, max_requests=25)

        assert [(f.requirement_id, f.kind) for f in report.findings] == [
            ("r0", FindingKind.CONFLICT),
            ("r5", FindingKind.MUST_INFLATION),
            ("r4", FindingKind.MUST_INFLATION),
        ]
        assert (report.conflicts, report.must_inflation, report.capped) == (1, 2, False)
        assert report.must_share == 5 / 6

    def test_teto_de_pedidos_e_declarado_e_preserva_os_conflitos(self):
        items = [_item(0, C, category="SE")] + [_item(i, M, 1.0 - i / 10) for i in range(1, 6)]

        report = mediate(items, Lang.PT, max_requests=1)

        assert [f.requirement_id for f in report.findings] == ["r0"]
        assert report.capped is True

    def test_sem_problemas_nao_ha_pedidos(self):
        report = mediate([_item(0, M), _item(1, S), _item(2, S)], Lang.PT)

        assert report.findings == [] and report.capped is False


class TestRequest:
    def test_pedido_tipado_do_mediador_ao_priorizador_com_a_evidencia(self):
        [finding] = find_conflicts([_item(0, C, category="SE")], Lang.EN)

        message = request_for(finding, Lang.EN)

        assert (message.sender, message.receiver) == (MEDIATOR, "prioritizer")
        assert message.performative is Performative.REQUEST and message.requirement_id == "r0"
        assert isinstance(message.content, ReprioritizationRequest)
        assert "conflicting" in message.content.reason
        assert "critical category 'SE'" in message.content.evidence

    def test_o_pedido_sai_no_idioma_da_execucao(self):
        [finding] = find_conflicts([_item(0, C, category="SE")], Lang.ES)

        content = request_for(finding, Lang.ES).content

        assert isinstance(content, ReprioritizationRequest)
        assert "contradictorias" in content.reason and "categoría crítica" in content.evidence
