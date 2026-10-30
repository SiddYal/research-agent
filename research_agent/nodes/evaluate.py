from __future__ import annotations

from research_agent.evaluation import evaluate
from research_agent.nodes.base import Deps, traced
from research_agent.state import ResearchState


@traced("evaluate")
def evaluate_node(state: ResearchState, deps: Deps) -> dict:
    # Totals (cost, latency) are finalized by the service once this node's own
    # timeline event exists; see evaluation.finalize_totals.
    return {"evaluation": evaluate(state, deps.llm)}
