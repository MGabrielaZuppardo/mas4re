from __future__ import annotations

from domain.agent_actions import (
    ActionKind,
    ActionRecord,
    AgentEpisode,
    Belief,
    EpisodeOutcome,
    Goal,
)


def _episode(actions: list[ActionRecord], **overrides) -> AgentEpisode:
    fields = {
        "agent": "classifier",
        "requirement_id": "r1",
        "model": "fake/model",
        "goal": Goal(description="classificar o requisito"),
        "actions": actions,
    }
    return AgentEpisode(**{**fields, **overrides})


def _action(step: int, kind: ActionKind, name: str, **overrides) -> ActionRecord:
    return ActionRecord(step=step, kind=kind, name=name, **overrides)


class TestAgentEpisode:
    def test_sem_acoes_o_episodio_nasce_como_orcamento_esgotado(self):
        episode = _episode([])

        assert episode.outcome is EpisodeOutcome.BUDGET_EXHAUSTED
        assert episode.submitted_via is None
        assert episode.llm_calls == 0

    def test_chamadas_de_llm_contam_passos_distintos_nao_acoes(self):
        episode = _episode(
            [
                _action(1, ActionKind.TOOL, "lookup_nfr_taxonomy", input_tokens=100),
                _action(1, ActionKind.TOOL, "recall_similar_cases"),
                _action(2, ActionKind.SUBMIT, "submit_classification", output_tokens=20),
            ]
        )

        assert episode.llm_calls == 2

    def test_tokens_somam_o_que_foi_atribuido_a_cada_acao(self):
        episode = _episode(
            [
                _action(1, ActionKind.TOOL, "a", input_tokens=100, output_tokens=10),
                _action(2, ActionKind.SUBMIT, "b", input_tokens=150, output_tokens=25),
            ]
        )

        assert (episode.input_tokens, episode.output_tokens) == (250, 35)

    def test_a_sequencia_de_acoes_ignora_as_invalidas(self):
        episode = _episode(
            [
                _action(1, ActionKind.INVALID, "ferramenta_inexistente"),
                _action(2, ActionKind.TOOL, "lookup_nfr_taxonomy"),
                _action(3, ActionKind.SUBMIT, "submit_classification"),
            ]
        )

        assert episode.action_sequence == ("lookup_nfr_taxonomy", "submit_classification")

    def test_objetivo_e_crencas_sobrevivem_a_serializacao(self):
        episode = _episode(
            [],
            beliefs=[Belief(source="percept", content="requisito: responder em 200ms")],
            outcome=EpisodeOutcome.SUBMITTED,
            submitted_via="tool",
        )

        restored = AgentEpisode.model_validate(episode.model_dump(mode="json"))

        assert restored.goal.description == "classificar o requisito"
        assert restored.beliefs[0].source == "percept"
        assert restored.outcome is EpisodeOutcome.SUBMITTED and restored.submitted_via == "tool"
