from __future__ import annotations

from pydantic import BaseModel

from agents.agentic.toolkit import (
    make_consistency_tool,
    make_recall_tool,
    make_terminal_tool,
    structural_issues,
)
from agents.memory import EpisodicMemory
from domain.enums import RequirementType

_CODES = ["PE", "SE", "US"]


class TestStructuralIssues:
    def test_funcional_sem_categoria_e_consistente(self):
        assert structural_issues(RequirementType.FUNCTIONAL, None, _CODES) == []

    def test_nao_funcional_com_categoria_conhecida_e_consistente(self):
        assert structural_issues(RequirementType.NON_FUNCTIONAL, "pe", _CODES) == []

    def test_nao_funcional_sem_categoria(self):
        [issue] = structural_issues(RequirementType.NON_FUNCTIONAL, None, _CODES)

        assert "needs an nfr_category" in issue

    def test_funcional_com_categoria(self):
        [issue] = structural_issues(RequirementType.FUNCTIONAL, "PE", _CODES)

        assert "must have nfr_category null" in issue

    def test_categoria_desconhecida_lista_os_codigos_validos(self):
        [issue] = structural_issues(RequirementType.NON_FUNCTIONAL, "ZZ", _CODES)

        assert "Unknown category 'ZZ'" in issue and "PE, SE, US" in issue

    def test_acumula_os_problemas(self):
        issues = structural_issues(RequirementType.FUNCTIONAL, "ZZ", _CODES)

        assert len(issues) == 2


class TestConsistencyTool:
    def test_resposta_para_o_agente(self):
        tool = make_consistency_tool(_CODES)

        assert tool.invoke({"requirement_type": "NF", "nfr_category": "SE"}) == "Consistent."
        assert tool.invoke({"requirement_type": "NF"}).startswith("Inconsistent:")


class TestRecallTool:
    def test_sem_memoria_ou_sem_parecidos(self):
        assert make_recall_tool(lambda: None).invoke({"query": "x"}).startswith("No similar")
        assert (
            make_recall_tool(lambda: EpisodicMemory())
            .invoke({"query": "x"})
            .startswith("No similar")
        )

    def test_lista_os_casos_parecidos_com_a_decisao(self):
        memory = EpisodicMemory()
        memory.add("a", "sistema deve autenticar usuários", "F/-", "login")

        result = make_recall_tool(lambda: memory).invoke({"query": "autenticar usuários novos"})

        assert "sistema deve autenticar usuários" in result and "F/-" in result

    def test_le_a_memoria_atual_a_cada_chamada(self):
        holder: list[EpisodicMemory | None] = [None]
        tool = make_recall_tool(lambda: holder[0])
        assert tool.invoke({"query": "autenticar usuários"}).startswith("No similar")

        holder[0] = EpisodicMemory()
        holder[0].add("a", "autenticar usuários", "F/-", "login")

        assert "autenticar usuários" in tool.invoke({"query": "autenticar usuários"})


class TestTerminalTool:
    def test_expoe_o_esquema_dos_argumentos(self):
        class Answer(BaseModel):
            label: str

        tool = make_terminal_tool("finish", "Ends the work.", Answer)

        assert tool.name == "finish" and tool.args_schema is Answer
        assert tool.invoke({"label": "x"}) == "Recorded."
