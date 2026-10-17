from __future__ import annotations

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from agents.agentic.runtime import AgentRuntime, EpisodeTask
from domain.agent_actions import ActionKind, Belief, EpisodeOutcome, Goal
from tests.helpers.scripted_llm import ScriptedChatModel, text_reply, tool_call, tool_calls


class _LookupArgs(BaseModel):
    term: str


class _SubmitArgs(BaseModel):
    label: str
    confidence: float


def _lookup(term: str) -> str:
    if term == "boom":
        raise RuntimeError("dictionary offline")
    return f"definition of {term}"


def _tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_lookup, name="lookup", description="Look a term up.", args_schema=_LookupArgs
        ),
        StructuredTool.from_function(
            func=lambda label, confidence: "ignored",
            name="submit",
            description="Submit the answer.",
            args_schema=_SubmitArgs,
        ),
    ]


def _runtime(script: list, max_steps: int = 5) -> tuple[AgentRuntime, ScriptedChatModel]:
    llm = ScriptedChatModel(script=script)
    return AgentRuntime(llm, _tools(), frozenset({"submit"}), max_steps=max_steps), llm


def _task(**overrides) -> EpisodeTask:
    fields = {
        "agent": "tester",
        "model": "fake/model",
        "requirement_id": "r1",
        "goal": Goal(description="rotular o item"),
        "system_prompt": "Você rotula itens.",
        "percept": "item: responder em 200ms",
    }
    return EpisodeTask(**{**fields, **overrides})


def _submit(label: str = "x", confidence: float = 0.9, call_id: str = "s1"):
    return tool_call("submit", call_id=call_id, label=label, confidence=confidence)


class TestSubmission:
    def test_agente_que_responde_de_primeira_termina_em_um_passo(self):
        runtime, llm = _runtime([_submit("NF")])

        result = runtime.run(_task())

        assert result.episode.outcome is EpisodeOutcome.SUBMITTED
        assert result.episode.submitted_via == "tool"
        assert result.submission is not None
        assert result.submission.action == "submit"
        assert result.submission.arguments.label == "NF"
        assert result.episode.llm_calls == 1 and llm.calls == 1

    def test_tokens_e_latencia_ficam_na_primeira_acao_da_chamada(self):
        runtime, _ = _runtime([_submit()])

        episode = runtime.run(_task()).episode

        assert (episode.input_tokens, episode.output_tokens) == (100, 10)
        assert episode.actions[0].latency_ms >= 0

    def test_o_modelo_ve_o_objetivo_e_o_percepto(self):
        runtime, llm = _runtime([_submit()])

        runtime.run(_task(system_prompt="OBJETIVO: rotular o item", percept="item: X"))

        system, user = llm.seen[0]
        assert "OBJETIVO: rotular o item" in system.content
        assert user.content == "item: X"

    def test_crencas_iniciais_entram_no_episodio(self):
        runtime, _ = _runtime([_submit()])

        episode = runtime.run(_task(beliefs=[Belief(source="percept", content="item X")])).episode

        assert episode.beliefs == [Belief(source="percept", content="item X")]
        assert episode.goal.description == "rotular o item"


class TestToolUse:
    def test_ferramenta_chamada_devolve_observacao_e_vira_crenca(self):
        runtime, llm = _runtime([tool_call("lookup", term="latency"), _submit()])

        episode = runtime.run(_task()).episode

        assert episode.action_sequence == ("lookup", "submit")
        assert episode.actions[0].observation == "definition of latency"
        assert Belief(source="tool:lookup", content="definition of latency") in episode.beliefs
        second_call = llm.seen[1]
        assert isinstance(second_call[-1], ToolMessage)
        assert second_call[-1].content == "definition of latency"

    def test_a_ferramenta_pode_ser_chamada_varias_vezes_a_escolha_e_do_modelo(self):
        runtime, _ = _runtime(
            [tool_call("lookup", term="a"), tool_call("lookup", term="b"), _submit()]
        )

        episode = runtime.run(_task()).episode

        assert episode.action_sequence == ("lookup", "lookup", "submit")

    def test_varias_chamadas_no_mesmo_passo_sao_todas_executadas(self):
        runtime, _ = _runtime(
            [tool_calls(("lookup", {"term": "a"}), ("lookup", {"term": "b"})), _submit()]
        )

        episode = runtime.run(_task()).episode

        assert [a.name for a in episode.actions] == ["lookup", "lookup", "submit"]
        assert [a.step for a in episode.actions] == [1, 1, 2]
        assert episode.actions[0].input_tokens == 100 and episode.actions[1].input_tokens == 0

    def test_apos_o_termino_as_chamadas_restantes_do_passo_sao_ignoradas(self):
        runtime, _ = _runtime(
            [tool_calls(("submit", {"label": "x", "confidence": 0.5}), ("lookup", {"term": "a"}))]
        )

        episode = runtime.run(_task()).episode

        assert episode.action_sequence == ("submit",)

    def test_ferramenta_que_falha_vira_acao_invalida_e_o_agente_segue(self):
        runtime, _ = _runtime([tool_call("lookup", term="boom"), _submit()])

        episode = runtime.run(_task()).episode

        assert episode.actions[0].kind is ActionKind.INVALID
        assert "dictionary offline" in episode.actions[0].observation
        assert episode.outcome is EpisodeOutcome.SUBMITTED


class TestTheModelDecides:
    """The loop follows the model: different replies, different path."""

    def test_modelos_com_escolhas_diferentes_percorrem_caminhos_diferentes(self):
        direct, _ = _runtime([_submit()])
        careful, _ = _runtime([tool_call("lookup", term="a"), _submit()])

        assert direct.run(_task()).episode.action_sequence == ("submit",)
        assert careful.run(_task()).episode.action_sequence == ("lookup", "submit")

    def test_a_observacao_muda_a_acao_seguinte(self):
        def react_to_observation(messages: list[BaseMessage]):
            last = messages[-1]
            if isinstance(last, ToolMessage) and "unknown" in str(last.content):
                return tool_call("lookup", term="synonym", call_id="c2")
            return _submit()

        runtime, _ = _runtime(
            [tool_call("lookup", term="unknown"), react_to_observation, _submit()],
            max_steps=5,
        )
        runtime_known, _ = _runtime([tool_call("lookup", term="latency"), react_to_observation])

        assert runtime.run(_task()).episode.action_sequence == ("lookup", "lookup", "submit")
        assert runtime_known.run(_task()).episode.action_sequence == ("lookup", "submit")


class TestInvalidActions:
    def test_ferramenta_inexistente_e_observacao_de_erro_e_o_agente_corrige(self):
        runtime, llm = _runtime([tool_call("inventada", x=1), _submit()])

        episode = runtime.run(_task()).episode

        assert episode.actions[0].kind is ActionKind.INVALID
        assert "Unknown tool 'inventada'" in episode.actions[0].observation
        assert "lookup" in llm.seen[1][-1].content
        assert episode.outcome is EpisodeOutcome.SUBMITTED

    def test_argumentos_fora_do_esquema_nao_encerram_o_episodio(self):
        runtime, _ = _runtime([tool_call("submit", label="x"), _submit()])

        episode = runtime.run(_task()).episode

        assert episode.actions[0].kind is ActionKind.INVALID
        assert "Invalid arguments for 'submit'" in episode.actions[0].observation
        assert episode.action_sequence == ("submit",)
        assert episode.outcome is EpisodeOutcome.SUBMITTED

    def test_resposta_em_texto_que_nao_e_resposta_recebe_um_lembrete(self):
        runtime, llm = _runtime([text_reply("hmm, deixe-me pensar"), _submit()])

        episode = runtime.run(_task()).episode

        assert episode.actions[0].kind is ActionKind.INVALID
        reminder = llm.seen[1][-1]
        assert isinstance(reminder, HumanMessage)
        assert "finish by calling submit" in reminder.content
        assert episode.outcome is EpisodeOutcome.SUBMITTED


class TestTextSubmission:
    def test_json_valido_no_texto_conta_como_resposta_e_e_marcado(self):
        runtime, _ = _runtime([text_reply('{"label": "F", "confidence": 0.8}')])

        result = runtime.run(_task())

        assert result.episode.outcome is EpisodeOutcome.SUBMITTED
        assert result.episode.submitted_via == "text"
        assert result.episode.actions[0].kind is ActionKind.SUBMIT
        assert result.submission is not None and result.submission.arguments.label == "F"

    def test_json_com_campos_invalidos_no_texto_nao_conta(self):
        runtime, _ = _runtime([text_reply('{"label": "F"}'), _submit()])

        episode = runtime.run(_task()).episode

        assert episode.actions[0].kind is ActionKind.INVALID
        assert episode.submitted_via == "tool"


class TestBudget:
    def test_orcamento_esgotado_nao_inventa_resposta(self):
        runtime, llm = _runtime([tool_call("lookup", term=str(i)) for i in range(3)], max_steps=3)

        result = runtime.run(_task())

        assert result.submission is None
        assert result.episode.outcome is EpisodeOutcome.BUDGET_EXHAUSTED
        assert result.episode.submitted_via is None
        assert llm.calls == 3

    def test_o_orcamento_nao_limita_quem_termina_antes(self):
        runtime, llm = _runtime([tool_call("lookup", term="a"), _submit()], max_steps=10)

        runtime.run(_task())

        assert llm.calls == 2


class TestConstruction:
    def test_ferramenta_terminal_precisa_estar_entre_as_ferramentas(self):
        with pytest.raises(ValueError, match="Terminal tools"):
            AgentRuntime(ScriptedChatModel(script=[]), _tools(), frozenset({"finish"}))

    def test_ferramenta_sem_esquema_pydantic_e_recusada(self):
        tool = StructuredTool.from_function(
            func=lambda: "x",
            name="sem_esquema",
            description="d",
            args_schema={"type": "object", "properties": {}},
        )

        with pytest.raises(ValueError, match="pydantic args_schema"):
            AgentRuntime(ScriptedChatModel(script=[]), [tool], frozenset({"sem_esquema"}))
