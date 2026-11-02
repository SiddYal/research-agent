from __future__ import annotations

from research_agent.nodes.base import Deps, traced
from research_agent.state import ResearchState
from research_agent.tools.retrieval import HybridRetriever, llm_rerank


def make_retriever(state: ResearchState, deps: Deps) -> HybridRetriever:
    return HybridRetriever(state.job_id, state.chunks, deps.embedder, deps.store)


@traced("retrieve")
def retrieve_node(state: ResearchState, deps: Deps) -> dict:
    s = deps.settings
    retriever = make_retriever(state, deps)
    retrieved = dict(state.retrieved)
    for sq in state.active_subquestions():
        query = " ".join([sq.question] + sq.search_queries[:1])
        candidates = retriever.search(query, s.retrieval_candidates)
        if s.reranker == "llm":
            top = llm_rerank(deps.llm, sq.question, candidates, s.top_k_chunks)
        else:
            top = candidates[: s.top_k_chunks]
        retrieved[sq.id] = [c.id for c in top]
    return {"retrieved": retrieved}
