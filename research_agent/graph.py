"""The LangGraph workflow.

    planner -> human_approval -(approved)-> search -> ingest -> retrieve -> extract_claims
            -> verify -(unresolved claims, rounds left)-> followup_search -> verify
                      -(otherwise)-> writer -> evaluate -> END

State is checkpointed after every node under thread_id = job id, so a job can be
paused at approval, inspected, and resumed, and a failed node can be retried
without re-running the nodes before it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from research_agent.config import Settings, get_settings
from research_agent.llm import BudgetExceeded
from research_agent.nodes import (
    Deps,
    approval_node,
    evaluate_node,
    extract_node,
    followup_node,
    ingest_node,
    planner_node,
    retrieve_node,
    search_node,
    verify_node,
    writer_node,
)
from research_agent.nodes.verify import needs_followup
from research_agent.state import ResearchState


def _retryable(exc: Exception) -> bool:
    # External calls already retry transient errors themselves; a node-level retry
    # is one more attempt for anything unexpected - never for a blown budget, a bug,
    # or a client error such as a bad API key (HTTP 4xx other than 429).
    if isinstance(exc, (BudgetExceeded, ValueError, KeyError, TypeError)):
        return False
    status = getattr(exc, "status_code", None)
    return not (isinstance(status, int) and 400 <= status < 500 and status != 429)


RETRY = RetryPolicy(max_attempts=2, initial_interval=2.0, retry_on=_retryable)


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver | None = None):
    g = StateGraph(ResearchState)
    nodes = {
        "planner": planner_node,
        "human_approval": approval_node,
        "search": search_node,
        "ingest": ingest_node,
        "retrieve": retrieve_node,
        "extract_claims": extract_node,
        "verify": verify_node,
        "followup_search": followup_node,
        "writer": writer_node,
        "evaluate": evaluate_node,
    }
    for name, fn in nodes.items():
        g.add_node(name, (lambda f: lambda state: f(state, deps))(fn),
                   retry_policy=None if name == "human_approval" else RETRY)

    g.add_edge(START, "planner")
    g.add_edge("planner", "human_approval")
    g.add_conditional_edges("human_approval", lambda s: "search" if s.approved else END, ["search", END])
    g.add_edge("search", "ingest")
    g.add_edge("ingest", "retrieve")
    g.add_edge("retrieve", "extract_claims")
    g.add_edge("extract_claims", "verify")
    g.add_conditional_edges(
        "verify",
        lambda s: "followup_search" if needs_followup(s, deps.settings.max_verification_rounds) else "writer",
        ["followup_search", "writer"],
    )
    g.add_edge("followup_search", "verify")
    g.add_edge("writer", "evaluate")
    g.add_edge("evaluate", END)
    return g.compile(checkpointer=checkpointer)


def _serde() -> JsonPlusSerializer:
    """Checkpoints may only deserialize into our own state models."""
    from pydantic import BaseModel

    from research_agent import state as state_module

    models = [
        obj for obj in vars(state_module).values()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == state_module.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=models)


def make_checkpointer(settings: Settings | None = None) -> BaseCheckpointSaver:
    settings = settings or get_settings()
    url = settings.checkpoint_url or settings.database_url
    if url.startswith("postgresql"):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            url.replace("postgresql+psycopg://", "postgresql://"),
            max_size=settings.worker_threads + 2,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        saver = PostgresSaver(pool, serde=_serde())
        saver.setup()
        return saver

    from langgraph.checkpoint.sqlite import SqliteSaver

    if settings.checkpoint_url:
        path = settings.checkpoint_url.removeprefix("sqlite:///")
    else:  # data/research.db -> data/research.checkpoints.db
        db = Path(url.removeprefix("sqlite:///"))
        path = str(db.with_suffix(".checkpoints.db")) if ":memory:" not in url else ":memory:"
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=_serde())
