from __future__ import annotations

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from research_agent.nodes.base import Deps, traced
from research_agent.state import Plan, ResearchState, SubQuestion


class _SubQ(BaseModel):
    question: str
    search_queries: list[str] = Field(description="2-3 distinct web search queries")


class _PlanOut(BaseModel):
    objective: str = Field(description="One or two sentences: what the report must establish")
    scope: str = Field(description="What is in and out of scope, incl. timeframe if relevant")
    outline: list[str] = Field(description="Report section titles, in order")
    subquestions: list[_SubQ]


PLANNER_SYSTEM = """You are an expert research planner. Turn the user's question into a research plan:
- an objective and scope,
- {min_q}-{max_q} non-overlapping subquestions that together fully answer the question,
- 2-3 concise, keyword-style web search queries per subquestion (vary the wording; include the
  current year when recency matters),
- a report outline whose sections follow the subquestions.
Today's date is {today}."""


@traced("planner")
def planner_node(state: ResearchState, deps: Deps) -> dict:
    from datetime import date

    max_q = deps.settings.max_subquestions
    out = deps.llm.structured(
        PLANNER_SYSTEM.format(min_q=min(3, max_q), max_q=max_q, today=date.today().isoformat()),
        f"Research question: {state.question}",
        _PlanOut,
        max_tokens=1500,
    )
    if not out.subquestions:
        raise ValueError("planner returned no subquestions")
    plan = Plan(
        objective=out.objective,
        scope=out.scope,
        outline=out.outline,
        subquestions=[
            SubQuestion(id=f"Q{i}", question=sq.question, search_queries=sq.search_queries[:3])
            for i, sq in enumerate(out.subquestions[:max_q], 1)
        ],
    )
    return {"plan": plan}


@traced("human_approval")
def approval_node(state: ResearchState, deps: Deps) -> dict:
    """Pauses the graph until a human approves (optionally editing) the plan.

    Resume with Command(resume={"approved": bool, "plan": <optional edited plan dict>}).
    """
    if state.auto_approve:
        return {"approved": True}
    decision = interrupt({"type": "plan_approval", "plan": state.plan.model_dump()})
    update: dict = {"approved": bool(decision.get("approved"))}
    if decision.get("plan"):
        edited = Plan.model_validate(decision["plan"])
        for i, sq in enumerate(edited.subquestions, 1):
            sq.id = f"Q{i}"
            if not sq.search_queries:
                sq.search_queries = [sq.question]
        update["plan"] = edited
    return update
