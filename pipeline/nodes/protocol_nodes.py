"""Nodes of the interaction protocols P2 and P3: the turns in which an agent answers what it was
asked, and the mediator's turn. Each does one thing and leaves the deciding to the agents."""

from __future__ import annotations

from agents.agentic.classifier import AgenticClassifier
from agents.agentic.prioritizer import AgenticPrioritizer
from domain.enums import Lang
from domain.models import PipelineState
from pipeline.mediation import MUST_SHARE_LIMIT, mediate, must_share, request_for


def classifier_answers_node(state: PipelineState, agent: AgenticClassifier) -> PipelineState:
    """P2: the classifier answers the reclassification requests the prioritizer made."""
    return agent.answer_requests(state)


def prioritizer_reconsiders_node(state: PipelineState, agent: AgenticPrioritizer) -> PipelineState:
    """P2: the prioritizer decides again, once, the items whose request was answered."""
    return agent.reconsider(state)


def mediator_node(
    state: PipelineState, lang: Lang, max_requests: int, must_limit: float = MUST_SHARE_LIMIT
) -> PipelineState:
    """P3: the mediator reads the finished backlog and sends the requests it justifies."""
    report = mediate(state.prioritized_requirements, lang, max_requests, must_limit)
    state.messages.extend(request_for(finding, lang) for finding in report.findings)
    state.metrics["mediator"] = {
        "conflicts": report.conflicts,
        "must_inflation": report.must_inflation,
        "requests": len(report.findings),
        "capped": report.capped,
        "must_limit": must_limit,
        "must_share_before": round(report.must_share, 4),
    }
    return state


def prioritizer_answers_node(state: PipelineState, agent: AgenticPrioritizer) -> PipelineState:
    """P3: the prioritizer answers the mediator's requests, agreeing or refusing each one."""
    agent.answer_requests(state)
    if "mediator" in state.metrics:
        state.metrics["mediator"]["must_share_after"] = round(
            must_share(state.prioritized_requirements), 4
        )
    return state
