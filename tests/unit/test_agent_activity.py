from __future__ import annotations

import pytest

from domain.agent_actions import (
    ActionKind,
    ActionRecord,
    AgentEpisode,
    EpisodeOutcome,
    Goal,
)
from evaluation.agent_activity import sequence_entropy, summarize_episodes


def _episode(
    names: list[tuple[ActionKind, str]],
    agent: str = "classifier",
    outcome: EpisodeOutcome = EpisodeOutcome.SUBMITTED,
    via: str | None = "tool",
) -> AgentEpisode:
    actions = [
        ActionRecord(step=i + 1, kind=kind, name=name, input_tokens=100, output_tokens=10)
        for i, (kind, name) in enumerate(names)
    ]
    return AgentEpisode(
        agent=agent,
        requirement_id="r",
        model="fake",
        goal=Goal(description="g"),
        actions=actions,
        outcome=outcome,
        submitted_via=via,
    )


_DIRECT = [(ActionKind.SUBMIT, "submit")]
_LOOKUP = [(ActionKind.TOOL, "lookup"), (ActionKind.SUBMIT, "submit")]


class TestSequenceEntropy:
    def test_todos_iguais_tem_entropia_zero(self):
        assert str(sequence_entropy([("a", "b")] * 5)) == "0.0"

    def test_metade_e_metade_tem_um_bit(self):
        assert sequence_entropy([("a",), ("a",), ("b",), ("b",)]) == 1.0

    def test_sem_sequencias_e_zero(self):
        assert sequence_entropy([]) == 0.0


class TestSummarizeEpisodes:
    def test_um_bloco_por_agente(self):
        summary = summarize_episodes([_episode(_DIRECT), _episode(_DIRECT, agent="prioritizer")])

        assert list(summary) == ["classifier", "prioritizer"]

    def test_termino_por_orcamento_e_por_texto(self):
        summary = summarize_episodes(
            [
                _episode(_DIRECT),
                _episode(_DIRECT, via="text"),
                _episode(_LOOKUP[:1], outcome=EpisodeOutcome.BUDGET_EXHAUSTED, via=None),
                _episode(_LOOKUP[:1], outcome=EpisodeOutcome.BUDGET_EXHAUSTED, via=None),
            ]
        )["classifier"]

        assert summary["episodes"] == 4 and summary["submitted"] == 2
        assert summary["budget_exhausted"] == 2 and summary["budget_exhaustion_rate"] == 0.5
        assert summary["submitted_via_text"] == 1

    def test_ferramentas_chamadas_por_nome_e_acoes_invalidas(self):
        summary = summarize_episodes(
            [
                _episode(_LOOKUP),
                _episode([(ActionKind.INVALID, ""), *_LOOKUP]),
            ]
        )["classifier"]

        assert summary["tool_calls"] == {"lookup": 2, "submit": 2}
        assert summary["invalid_actions"] == 1
        assert summary["invalid_action_rate"] == pytest.approx(1 / 5, abs=1e-4)

    def test_entropia_das_sequencias_distingue_agente_de_fluxo_fixo(self):
        fixed = summarize_episodes([_episode(_LOOKUP)] * 4)["classifier"]
        varied = summarize_episodes([_episode(_LOOKUP), _episode(_DIRECT)] * 2)["classifier"]

        assert fixed["action_sequence_entropy_bits"] == 0.0
        assert fixed["distinct_action_sequences"] == 1
        assert varied["action_sequence_entropy_bits"] == 1.0
        assert varied["distinct_action_sequences"] == 2

    def test_chamadas_de_llm_e_tokens(self):
        summary = summarize_episodes([_episode(_LOOKUP), _episode(_DIRECT)])["classifier"]

        assert summary["llm_calls"] == 3 and summary["mean_llm_calls"] == 1.5
        assert (summary["input_tokens"], summary["output_tokens"]) == (300, 30)

    def test_sem_episodios_devolve_vazio(self):
        assert summarize_episodes([]) == {}
