"""experiments/analyze_agency.py on synthetic runs: C1/C2/C4 read from the logs, C5 as an
association test, C7 as a paired comparison with the workflow. No LLM, no real run."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agents.agentic.toolkit import BacklogView
from agents.memory import EpisodicMemory
from domain.agent_actions import (
    ActionKind,
    ActionRecord,
    AgentEpisode,
    EpisodeOutcome,
    EpisodePhase,
    Goal,
)
from domain.enums import InformationRegime, RequirementType
from domain.models import PrioritizedRequirement
from experiments import analyze_agency
from experiments.analysis_common import MixedInformationRegimeError, write_analysis_result
from experiments.analyze_agency import (
    PRIMARY_CELL,
    Cell,
    MixedCellError,
    UnpairableRunsError,
    agent_criteria,
    analyze,
    association,
    compare_to_workflow,
    load_run,
    observation_action_association,
    observations,
    single_cell,
    single_regime,
)

_N = 24
_GOLD_CYCLE = ["M", "S", "S", "M", "C", "S"]


# ── synthetic runs ─────────────────────────────────────────────────────────────────────


def _prediction(
    index: int, type_correct: bool = True, priority: str | None = None
) -> dict[str, Any]:
    label = "F" if index % 2 == 0 else "NF"
    predicted = label if type_correct else ("NF" if label == "F" else "F")
    item = PrioritizedRequirement(
        id=f"{index + 1:04d}",
        text=f"Requisito {index}",
        requirement_type=RequirementType(predicted),
        nfr_category=None if predicted == "F" else "US",
        confidence=0.9,
        priority=priority,
        priority_score=0.5,
        priority_rank=index + 1,
        metadata={"label_type": label, "ground_truth_priority": _GOLD_CYCLE[index % 6]},
    )
    return item.model_dump(mode="json")


def _action(
    name: str,
    kind: ActionKind = ActionKind.TOOL,
    observation: str = "",
    step: int = 1,
    **arguments: Any,
) -> ActionRecord:
    return ActionRecord(
        step=step, kind=kind, name=name, arguments=arguments, observation=observation
    )


def _episode(
    actions: list[ActionRecord],
    agent: str = "classifier",
    phase: EpisodePhase = EpisodePhase.WORK,
) -> AgentEpisode:
    return AgentEpisode(
        agent=agent,
        requirement_id="0001",
        model="ollama/fake",
        goal=Goal(description="classify"),
        actions=actions,
        outcome=EpisodeOutcome.SUBMITTED,
        phase=phase,
    )


def _write_run(
    root: Path,
    name: str,
    strategy: str,
    predictions: list[dict[str, Any]],
    episodes: list[AgentEpisode] | None = None,
    metrics: dict[str, Any] | None = None,
    **manifest: Any,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    full_manifest = {
        "strategy": strategy,
        "model": "ollama/fake+ollama/fake",
        "lang": "es",
        "data_lang": "es",
        "dataset_name": "priorespa",
        "dataset": "datasets/data/priorespa/PrioReSpa.csv",
        "dataset_md5": "abc",
        "dataset_n": _N,
        "seed": 42,
        "temperature": 0.0,
        "elapsed_seconds": 120.0,
        "information_regime": "zero_shot",
        **manifest,
    }
    results = {
        "config": {"strategy": strategy},
        "metrics": metrics or {},
        "n_failed": _N - len(predictions),
        "predictions": predictions,
    }
    (run_dir / "manifest.json").write_text(json.dumps(full_manifest), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    if episodes:
        lines = [episode.model_dump_json() for episode in episodes]
        (run_dir / "actions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def _workflow_and_agentic(root: Path, **agentic_manifest: Any):
    """Workflow answers Must for everything and misses 6 types; the agentic run follows the gold,
    misses 3 types (one the workflow got right) and fails to process 2 items."""
    workflow_wrong_types = {0, 1, 2, 3, 4, 5}
    agentic_wrong_types = {0, 1, 6}
    workflow = [_prediction(i, i not in workflow_wrong_types, priority="M") for i in range(_N)]
    agentic = [
        _prediction(i, i not in agentic_wrong_types, priority=_GOLD_CYCLE[i % 6])
        for i in range(_N - 2)
    ]
    workflow_dir = _write_run(root, "wf", "pipeline", workflow, elapsed_seconds=100.0)
    agentic_dir = _write_run(
        root,
        "ag",
        "pipeline_agentic_p2",
        agentic,
        metrics={
            "agent_activity": {
                "classifier": {"llm_calls": 30, "input_tokens": 1000, "output_tokens": 200},
                "prioritizer": {"llm_calls": 42, "input_tokens": 800, "output_tokens": 100},
            }
        },
        protocol="p2",
        elapsed_seconds=300.0,
        **agentic_manifest,
    )
    return load_run(workflow_dir), load_run(agentic_dir)


# ── loading and refusing to mix ────────────────────────────────────────────────────────


class TestLoading:
    def test_le_manifesto_resultados_e_episodios(self, tmp_path):
        directory = _write_run(
            tmp_path,
            "run",
            "pipeline_agentic",
            [_prediction(0, priority="M")],
            episodes=[_episode([_action("submit_classification", ActionKind.SUBMIT)])],
        )

        run = load_run(directory)

        assert run.strategy == "pipeline_agentic" and list(run.items) == ["0001"]
        assert len(run.episodes) == 1 and run.cell == PRIMARY_CELL

    def test_execucao_sem_actions_jsonl_tem_zero_episodios(self, tmp_path):
        run = load_run(_write_run(tmp_path, "wf", "pipeline", [_prediction(0)]))

        assert run.episodes == []

    def test_manifesto_anterior_as_celulas_de_idioma_e_recusado(self, tmp_path):
        directory = _write_run(tmp_path, "old", "pipeline", [_prediction(0)])
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        del manifest["data_lang"]
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="data_lang"):
            load_run(directory)


class TestCells:
    def test_so_es_es_no_priorespa_e_primaria(self):
        assert Cell("priorespa", "es", "es").status == "primary"
        assert Cell("priorespa", "es", "pt").status == "sensitivity"
        assert Cell("priorespa", "pt", "pt").status == "sensitivity"
        assert Cell("promise", "pt", "pt").status == "sensitivity"

    def test_celulas_diferentes_nao_se_misturam(self, tmp_path):
        es = load_run(_write_run(tmp_path, "a", "pipeline_agentic", [_prediction(0)]))
        pt = load_run(_write_run(tmp_path, "b", "pipeline_agentic", [_prediction(0)], lang="pt"))

        with pytest.raises(MixedCellError, match="priorespa/es/pt"):
            single_cell([es, pt])

    def test_regimes_de_informacao_diferentes_nao_se_misturam(self, tmp_path):
        zero = load_run(_write_run(tmp_path, "a", "pipeline", [_prediction(0)]))
        hybrid = load_run(
            _write_run(
                tmp_path,
                "b",
                "pipeline",
                [_prediction(0)],
                information_regime=InformationRegime.HYBRID_EXPLORATORY.value,
            )
        )

        with pytest.raises(MixedInformationRegimeError):
            single_regime([zero, hybrid])

    def test_workflow_so_se_compara_com_agentico_que_difere_apenas_na_estrategia(self, tmp_path):
        workflow, other_seed = _workflow_and_agentic(tmp_path, seed=7)

        with pytest.raises(UnpairableRunsError, match="seed: 42 != 7"):
            compare_to_workflow(workflow, other_seed)


# ── C1, C2, C4 ─────────────────────────────────────────────────────────────────────────


class TestAgentCriteria:
    @staticmethod
    def _run(tmp_path: Path, episodes: list[AgentEpisode], interaction: dict | None = None):
        metrics = {"interaction": interaction} if interaction else {}
        directory = _write_run(
            tmp_path, "run", "pipeline_agentic_p2", [_prediction(0)], episodes, metrics
        )
        return load_run(directory)

    def test_entropia_positiva_quando_os_episodios_seguem_caminhos_diferentes(self, tmp_path):
        direct = _episode([_action("submit_classification", ActionKind.SUBMIT)])
        careful = _episode(
            [_action("check_consistency"), _action("submit_classification", ActionKind.SUBMIT)]
        )

        criteria = agent_criteria(self._run(tmp_path, [direct, careful]))

        block = criteria["c1_c2"]["classifier"]
        assert block["entropy_above_zero"] is True and block["distinct_action_sequences"] == 2

    def test_entropia_zero_quando_todos_seguem_a_mesma_sequencia(self, tmp_path):
        same = [_episode([_action("submit_classification", ActionKind.SUBMIT)])] * 3

        block = agent_criteria(self._run(tmp_path, same))["c1_c2"]["classifier"]

        assert block["entropy_above_zero"] is False

    @pytest.mark.parametrize(
        ("agree_rate", "refuse_rate", "expected"),
        [(0.6, 0.4, True), (1.0, 0.0, False), (None, None, None)],
    )
    def test_c4_exige_aceite_e_recusa_ambos_acima_de_zero(
        self, tmp_path, agree_rate, refuse_rate, expected
    ):
        interaction = {"requests": 5, "agree_rate": agree_rate, "refuse_rate": refuse_rate}
        episode = _episode([_action("submit_classification", ActionKind.SUBMIT)])

        criteria = agent_criteria(self._run(tmp_path, [episode], interaction))

        assert criteria["c4"]["both_rates_positive"] is expected

    def test_sem_mensagens_registradas_c4_nao_tem_veredito(self, tmp_path):
        episode = _episode([_action("submit_classification", ActionKind.SUBMIT)])

        assert agent_criteria(self._run(tmp_path, [episode]))["c4"]["both_rates_positive"] is None


# ── C5 ─────────────────────────────────────────────────────────────────────────────────


def _checked(observation: str, checked: dict, submitted: dict) -> AgentEpisode:
    return _episode(
        [
            _action("check_consistency", observation=observation, step=1, **checked),
            _action("submit_classification", ActionKind.SUBMIT, step=2, **submitted),
        ]
    )


_NF_SE = {"requirement_type": "NF", "nfr_category": "SE"}
_F = {"requirement_type": "F", "nfr_category": None}


class TestObservations:
    def test_consistente_e_envio_da_mesma_resposta(self):
        [seen] = observations([_checked("Consistent.", _NF_SE, _NF_SE)])

        assert (seen.agent, seen.tool) == ("classifier", "check_consistency")
        assert (seen.observed, seen.followed) == ("consistent", "submits_checked_answer")

    def test_inconsistente_e_envio_de_outra_resposta(self):
        bad = {"requirement_type": "NF", "nfr_category": None}

        [seen] = observations([_checked("Inconsistent: needs a category.", bad, _NF_SE)])

        assert (seen.observed, seen.followed) == ("inconsistent", "submits_different_answer")

    def test_a_comparacao_ignora_caixa_e_nulo(self):
        [seen] = observations(
            [_checked("Consistent.", {"requirement_type": "nf", "nfr_category": "se"}, _NF_SE)]
        )

        assert seen.followed == "submits_checked_answer"

    def test_recall_vazio_ou_com_casos_e_a_acao_seguinte(self):
        empty = _episode(
            [
                _action("recall_similar_cases", observation="No similar cases handled so far."),
                _action("submit_classification", ActionKind.SUBMIT, step=2, **_F),
            ]
        )
        found = _episode(
            [
                _action("recall_similar_cases", observation='- "x" -> NF (ok)'),
                _action("lookup_nfr_taxonomy", step=2, category_code="SE"),
                _action("submit_classification", ActionKind.SUBMIT, step=3, **_F),
            ]
        )

        seen = observations([empty, found])

        assert [(o.observed, o.followed) for o in seen] == [
            ("no_similar_cases", "submits"),
            ("similar_cases_found", "calls_another_tool"),
        ]

    def test_visao_geral_le_a_parcela_de_musts_do_que_o_priorizador_realmente_ve(self):
        view = BacklogView(
            by_type=Counter({"F": 4}), by_category=Counter(), memory=EpisodicMemory()
        )
        episodes = []
        for decided, submitted in (([], "S"), (["M", "M", "S"], "S"), (["S", "S", "M"], "M")):
            view.decided = Counter(decided)
            episodes.append(
                _episode(
                    [
                        _action("backlog_overview", observation=view.render()),
                        _action("submit_priority", ActionKind.SUBMIT, step=2, priority=submitted),
                    ],
                    agent="prioritizer",
                )
            )

        seen = observations(episodes)

        assert [(o.observed, o.followed) for o in seen] == [
            ("no_priorities_yet", "submits_not_must"),
            ("must_share_high", "submits_not_must"),
            ("must_share_low", "submits_must"),
        ]

    def test_acoes_invalidas_entre_a_observacao_e_a_resposta_nao_contam_como_seguinte(self):
        episode = _episode(
            [
                _action("check_consistency", observation="Consistent.", **_NF_SE),
                _action("nao_existe", ActionKind.INVALID, step=2),
                _action("submit_classification", ActionKind.SUBMIT, step=3, **_NF_SE),
            ]
        )

        [seen] = observations([episode])

        assert seen.followed == "submits_checked_answer"

    def test_so_o_trabalho_inicial_conta_respostas_e_reconsideracoes_ficam_de_fora(self):
        actions = [
            _action("recall_similar_cases", observation="No similar cases handled so far."),
            _action("submit_priority", ActionKind.SUBMIT, step=2, priority="S"),
        ]

        seen = observations(
            [
                _episode(actions, "prioritizer", EpisodePhase.REPLY),
                _episode(actions, "prioritizer", EpisodePhase.RECONSIDERATION),
            ]
        )

        assert seen == []

    def test_observacao_sem_acao_seguinte_nao_e_contada(self):
        episode = _episode([_action("check_consistency", observation="Consistent.", **_NF_SE)])

        assert observations([episode]) == []

    def test_ferramentas_que_nao_observam_nao_geram_linhas(self):
        episode = _episode(
            [
                _action("lookup_nfr_taxonomy", observation="Security", category_code="SE"),
                _action("submit_classification", ActionKind.SUBMIT, step=2, **_NF_SE),
            ]
        )

        assert observations([episode]) == []


def _table(**rows: dict[str, int]) -> dict[str, Counter[str]]:
    return {name: Counter(counts) for name, counts in rows.items()}


class TestAssociation:
    def test_acao_que_depende_da_observacao_e_associada(self):
        report = association(
            _table(
                consistent={"submits_checked_answer": 18, "submits_different_answer": 2},
                inconsistent={"submits_checked_answer": 3, "submits_different_answer": 17},
            )
        )

        assert report["test"] == "fisher_exact" and report["verdict"] == "associated"
        assert report["p_value"] < 0.001 and report["cramers_v"] > 0.6 and report["n"] == 40

    def test_acao_independente_da_observacao_nao_e_associada(self):
        report = association(
            _table(
                consistent={"submits_checked_answer": 10, "submits_different_answer": 10},
                inconsistent={"submits_checked_answer": 10, "submits_different_answer": 10},
            )
        )

        assert report["verdict"] == "not_associated" and report["cramers_v"] == 0.0

    def test_poucas_observacoes_nao_dao_veredito(self):
        report = association(
            _table(
                consistent={"submits": 5, "calls_another_tool": 1},
                inconsistent={"submits": 1, "calls_another_tool": 5},
            )
        )

        assert report["verdict"] == "insufficient_data" and report["p_value"] is None
        assert "fewer than" in report["reason"]

    def test_uma_unica_classe_observada_nao_da_veredito(self):
        report = association(_table(consistent={"submits": 30, "calls_another_tool": 5}))

        assert report["verdict"] == "insufficient_data"
        assert "two observed classes" in report["reason"]

    def test_uma_unica_acao_possivel_nao_da_veredito(self):
        report = association(_table(consistent={"submits": 15}, inconsistent={"submits": 15}))

        assert report["verdict"] == "insufficient_data"

    def test_tabela_maior_que_2x2_usa_qui_quadrado_quando_as_contagens_permitem(self):
        report = association(
            _table(
                no_priorities_yet={
                    "submits_must": 20,
                    "submits_not_must": 5,
                    "calls_another_tool": 5,
                },
                must_share_low={"submits_must": 5, "submits_not_must": 20, "calls_another_tool": 5},
                must_share_high={
                    "submits_must": 5,
                    "submits_not_must": 5,
                    "calls_another_tool": 20,
                },
            )
        )

        assert report["test"] == "chi2" and report["verdict"] == "associated"

    def test_tabela_maior_que_2x2_com_esperados_pequenos_nao_da_veredito(self):
        report = association(
            _table(
                a={"x": 18, "y": 1, "z": 0},
                b={"x": 1, "y": 18, "z": 1},
                c={"x": 0, "y": 1, "z": 2},
            )
        )

        assert report["verdict"] == "insufficient_data" and "expected counts" in report["reason"]

    def test_associacao_por_agente_e_ferramenta(self):
        episodes = (
            [_checked("Consistent.", _NF_SE, _NF_SE)] * 18
            + [_checked("Consistent.", _NF_SE, _F)] * 2
            + [_checked("Inconsistent: x", _F, _NF_SE)] * 17
            + [_checked("Inconsistent: x", _F, _F)] * 3
        )

        report = observation_action_association(episodes)

        assert report["classifier"]["check_consistency"]["verdict"] == "associated"


# ── C7 ─────────────────────────────────────────────────────────────────────────────────


class TestCompareToWorkflow:
    @pytest.fixture
    def comparison(self, tmp_path):
        workflow, agentic = _workflow_and_agentic(tmp_path)
        return compare_to_workflow(workflow, agentic, n_bootstrap=300)

    def test_pareia_por_id_e_conta_o_que_so_um_dos_lados_processou(self, comparison):
        assert comparison["n_paired"] == _N - 2
        assert (comparison["only_workflow"], comparison["only_agentic"]) == (2, 0)
        assert comparison["protocol"] == "p2"

    def test_tipo_acuracia_diferenca_e_mcnemar_nos_itens_pareados(self, comparison):
        block = comparison["type_accuracy"]

        # the 22 paired items: the workflow misses 6 types, the agentic 3 (ids 0, 1 and 6)
        assert block["workflow"] == round(16 / 22, 4)
        assert block["agentic"] == round(19 / 22, 4)
        assert block["delta"] == round(3 / 22, 6)
        assert block["mcnemar"]["only_agentic_correct"] == 4
        assert block["mcnemar"]["only_workflow_correct"] == 1
        assert block["ci95"][0] <= block["delta"] <= block["ci95"][1]

    def test_acuracia_estrita_conta_o_item_nao_processado_como_erro(self, comparison):
        strict = comparison["type_accuracy"]["strict"]

        # workflow: 18 of 24 right; agentic: 19 of 24 (the 2 unprocessed items count as wrong)
        assert strict == {"workflow": round(18 / 24, 4), "agentic": round(19 / 24, 4)}

    def test_moscow_acuracia_e_kappa_ponderado_a_favor_de_quem_segue_o_gabarito(self, comparison):
        moscow = comparison["moscow"]

        assert moscow["n_paired"] == _N - 2
        assert moscow["accuracy"]["agentic"] == 1.0
        assert moscow["accuracy"]["workflow"] < 0.5
        assert moscow["accuracy"]["delta"] > 0.5
        assert moscow["weighted_kappa"]["delta"] > 0
        low, high = moscow["weighted_kappa"]["ci95"]
        assert low <= moscow["weighted_kappa"]["delta"] <= high

    def test_custo_e_cobertura_ficam_ao_lado_da_qualidade(self, comparison):
        cost = comparison["cost"]

        assert cost["agentic"]["coverage"] == round(22 / 24, 4)
        assert cost["agentic"]["n_failed"] == 2 and cost["workflow"]["n_failed"] == 0
        assert cost["workflow"]["seconds_per_item"] == round(100 / 24, 3)
        assert cost["seconds_ratio"] == 3.0
        assert cost["agentic"]["llm_calls_per_item"] == 3.0
        assert cost["agentic"]["tokens_per_item"] == round(2100 / 24, 1)
        assert cost["workflow"]["llm_calls_per_item"] is None

    def test_com_menos_de_dois_itens_pareados_nao_ha_veredito(self, tmp_path):
        workflow = load_run(_write_run(tmp_path, "wf", "pipeline", [_prediction(0, priority="M")]))
        agentic = load_run(
            _write_run(tmp_path, "ag", "pipeline_agentic", [_prediction(0, priority="M")])
        )

        comparison = compare_to_workflow(workflow, agentic, n_bootstrap=50)

        assert comparison["type_accuracy"]["verdict"] == "insufficient_data"
        assert comparison["moscow"]["verdict"] == "insufficient_data"


# ── the whole analysis and the command line ────────────────────────────────────────────


class TestAnalyze:
    def test_sem_workflow_so_ha_criterios_e_c5(self, tmp_path):
        episode = _episode([_action("submit_classification", ActionKind.SUBMIT)])
        run = load_run(_write_run(tmp_path, "ag", "pipeline_agentic", [_prediction(0)], [episode]))

        result = analyze([run])

        assert result["cell"]["status"] == "primary" and result["c7"] == []
        assert result["runs"][0]["strategy"] == "pipeline_agentic"
        assert "c1_c2" in result["runs"][0] and "c4" in result["runs"][0]

    def test_workflow_de_outra_celula_e_recusado(self, tmp_path):
        agentic = load_run(_write_run(tmp_path, "ag", "pipeline_agentic", [_prediction(0)]))
        workflow = load_run(
            _write_run(tmp_path, "wf", "pipeline", [_prediction(0)], data_lang="pt")
        )

        with pytest.raises(MixedCellError):
            analyze([agentic], workflow)

    def test_varias_execucoes_agenticas_viram_varias_comparacoes(self, tmp_path):
        workflow, p2 = _workflow_and_agentic(tmp_path)
        p3 = load_run(
            _write_run(
                tmp_path,
                "p3",
                "pipeline_agentic_p3",
                [_prediction(i, priority=_GOLD_CYCLE[i % 6]) for i in range(_N)],
                protocol="p3",
            )
        )

        result = analyze([p2, p3], workflow, n_bootstrap=50)

        assert [c["protocol"] for c in result["c7"]] == ["p2", "p3"]


class TestMain:
    def test_grava_o_resultado_com_manifesto_e_imprime_o_resumo(self, tmp_path, capsys):
        workflow, agentic = _workflow_and_agentic(tmp_path)
        out_dir = tmp_path / "analysis"

        def write_to_tmp(name, manifest, results):
            return write_analysis_result(name, manifest, results, out_dir=out_dir)

        with patch.object(analyze_agency, "write_analysis_result", write_to_tmp):
            analyze_agency.main(
                [
                    "--agentic",
                    str(agentic.path),
                    "--workflow",
                    str(workflow.path),
                    "--bootstrap",
                    "50",
                ]
            )

        [written] = out_dir.glob("agency_*.json")
        payload = json.loads(written.read_text(encoding="utf-8"))
        assert payload["manifest"]["script"] == "analyze_agency"
        assert payload["manifest"]["dataset"].endswith("PrioReSpa.csv")
        assert payload["results"]["cell"]["status"] == "primary"
        assert "cell: priorespa/es/es (primary)" in capsys.readouterr().out

    def test_workflow_precisa_ser_uma_execucao_do_pipeline(self, tmp_path):
        _, agentic = _workflow_and_agentic(tmp_path)

        with pytest.raises(SystemExit, match="'pipeline' run"):
            analyze_agency.main(["--agentic", str(agentic.path), "--workflow", str(agentic.path)])
