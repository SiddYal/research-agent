"""Follow-up questions on a finished report: plan the extra research and get it approved.

A follow-up re-enters the same graph and checkpoint thread as the original job. Its new
subquestions are appended to the plan and marked `active`, so search, retrieval,
extraction and verification only work on them; the writer then edits the existing
report instead of writing a new one (see writer.revise_report).
"""

from __future__ import annotations

from datetime import date

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from research_agent.evaluation import report_body
from research_agent.nodes.base import Deps, traced
from research_agent.nodes.planner import _SubQ
from research_agent.state import ResearchState, SubQuestion


class _FollowUpPlanOut(BaseModel):
    subquestions: list[_SubQ]


FOLLOWUP_SYSTEM = """You plan extra research for a reader's follow-up question about an existing report.
- Write 1-{max_q} subquestions that cover only what the follow-up needs and the report doesn't
  already establish. A narrow clarifying question usually needs just one.
- Don't repeat the research subquestions already covered.
- Give 2-3 concise, keyword-style web search queries per subquestion (include the current year
  when recency matters).
Today's date is {today}."""


def section_titles(report: str) -> list[str]:
    return [line[3:].strip() for line in report_body(report).splitlines() if line.startswith("## ")]


def _next_ids(state: ResearchState, n: int) -> list[str]:
    taken = {sq.id for sq in state.plan.subquestions}
    ids, i = [], len(state.plan.subquestions) + 1
    while len(ids) < n:
        if f"Q{i}" not in taken:
            ids.append(f"Q{i}")
        i += 1
    return ids


def _with_followup(state: ResearchState, **changes) -> list:
    followups = list(state.followups)
    followups[-1] = followups[-1].model_copy(update=changes)
    return followups


@traced("followup_planner")
def followup_planner_node(state: ResearchState, deps: Deps) -> dict:
    fu = state.pending_followup()
    max_q = deps.settings.followup_max_subquestions
    out = deps.llm.structured(
        FOLLOWUP_SYSTEM.format(max_q=max_q, today=date.today().isoformat()),
        f"Original research question: {state.question}\n"
        f"Report sections: {'; '.join(section_titles(state.report or ''))}\n"
        f"Already researched: {'; '.join(sq.question for sq in state.plan.subquestions)}\n\n"
        f"Follow-up question: {fu.question}",
        _FollowUpPlanOut,
        max_tokens=1000,
    )
    if not out.subquestions:
        raise ValueError("follow-up planner returned no subquestions")
    picked = out.subquestions[:max_q]
    new = [SubQuestion(id=i, question=sq.question, search_queries=sq.search_queries[:3])
           for i, sq in zip(_next_ids(state, len(picked)), picked)]
    plan = state.plan.model_copy(update={"subquestions": state.plan.subquestions + new})
    ids = [sq.id for sq in new]
    return {"plan": plan, "active": ids, "verification_round": 0,
            "followups": _with_followup(state, subquestion_ids=ids, status="awaiting_approval")}


@traced("followup_approval")
def followup_approval_node(state: ResearchState, deps: Deps) -> dict:
    """Pauses for the reader to approve, edit or reject the follow-up's subquestions.

    Resume with Command(resume={"approved": bool, "plan": {"subquestions": [...]}}), the same
    shape as the main plan approval; only the subquestions of an edited plan are used.
    """
    fu = state.pending_followup()
    if state.auto_approve:
        return {"followups": _with_followup(state, status="running")}
    new = [sq for sq in state.plan.subquestions if sq.id in fu.subquestion_ids]
    decision = interrupt({"type": "followup_approval", "question": fu.question,
                          "subquestions": [sq.model_dump() for sq in new]})
    kept = [sq for sq in state.plan.subquestions if sq.id not in fu.subquestion_ids]

    if not decision.get("approved"):
        return {"plan": state.plan.model_copy(update={"subquestions": kept}), "active": [],
                "followups": _with_followup(state, status="rejected", subquestion_ids=[],
                                            summary="You rejected the plan, so the report wasn't changed.")}
    edited = (decision.get("plan") or {}).get("subquestions")
    if edited:
        state_kept = state.model_copy(update={"plan": state.plan.model_copy(update={"subquestions": kept})})
        new = [SubQuestion(id=i, question=sq["question"], search_queries=sq.get("search_queries") or [sq["question"]])
               for i, sq in zip(_next_ids(state_kept, len(edited)), edited)]
    ids = [sq.id for sq in new]
    return {"plan": state.plan.model_copy(update={"subquestions": kept + new}), "active": ids,
            "followups": _with_followup(state, status="running", subquestion_ids=ids)}
