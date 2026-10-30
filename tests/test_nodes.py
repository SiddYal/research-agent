"""Each LangGraph node tested in isolation against a fixed ResearchState."""

import pytest

from research_agent.evaluation import cited_source_ids, evaluate, statements
from research_agent.nodes import (
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
from research_agent.nodes.verify import decide_status, needs_followup
from research_agent.state import Claim, Evidence, ResearchState, Source
from tests.fakes import FALSE_FACT, INJECTION, FakeSearch


def advance(state: ResearchState, *nodes, deps) -> ResearchState:
    for node in nodes:
        state = state.model_copy(update=node(state, deps))
    return state


@pytest.fixture
def planned(state, deps):
    return advance(state.model_copy(update={"auto_approve": True}), planner_node, approval_node, deps=deps)


@pytest.fixture
def ingested(planned, deps):
    return advance(planned, search_node, ingest_node, deps=deps)


@pytest.fixture
def verified(ingested, deps):
    return advance(ingested, retrieve_node, extract_node, verify_node, deps=deps)


def test_planner_builds_plan_with_ids(state, deps):
    update = planner_node(state, deps)
    plan = update["plan"]
    assert [q.id for q in plan.subquestions] == ["Q1", "Q2"]
    assert all(q.search_queries for q in plan.subquestions)
    assert update["timeline"][-1].node == "planner" and update["timeline"][-1].success


def test_approval_auto_and_edited_plan(planned, deps, monkeypatch):
    assert planned.approved is True
    # With a human in the loop, the node returns whatever the resume value says.
    import research_agent.nodes.planner as planner_mod

    edited = planned.plan.model_dump()
    edited["subquestions"] = [{"id": "x", "question": "Only this?", "search_queries": []}]
    monkeypatch.setattr(planner_mod, "interrupt", lambda payload: {"approved": True, "plan": edited})
    update = approval_node(planned.model_copy(update={"auto_approve": False}), deps)
    assert update["plan"].subquestions[0].id == "Q1"
    assert update["plan"].subquestions[0].search_queries == ["Only this?"]


def test_search_dedupes_and_tolerates_failed_queries(planned, deps):
    deps.search = FakeSearch(fail_queries={"eiffel tower height"})
    update = search_node(planned, deps)
    urls = [s.url for s in update["sources"]]
    assert len(urls) == len(set(urls)) == 3
    assert [s.id for s in update["sources"]] == ["S1", "S2", "S3"]
    assert any("search failed" in e for e in update["errors"])


def test_ingest_strips_planted_injection(ingested):
    planted = next(s for s in ingested.sources if "planted" in s.url)
    assert planted.injection_flags, "the planted injection should be flagged"
    assert not any(INJECTION in c.text for c in ingested.chunks)
    assert all(s.fetched for s in ingested.sources)
    assert {c.source_id for c in ingested.chunks} == {"S1", "S2", "S3"}


def test_retrieve_returns_ranked_chunks_per_subquestion(ingested, deps):
    update = retrieve_node(ingested, deps)
    assert set(update["retrieved"]) == {"Q1", "Q2"}
    known = {c.id for c in ingested.chunks}
    assert all(cid in known for ids in update["retrieved"].values() for cid in ids)
    assert len(update["retrieved"]["Q1"]) <= deps.settings.top_k_chunks


def test_extract_drops_claims_with_unknown_chunks(ingested, deps):
    st = advance(ingested, retrieve_node, deps=deps)
    st.retrieved["Q1"] = st.retrieved["Q1"] + ["S99-c0"]
    claims = extract_node(st, deps)["extracted_claims"]
    assert claims and all(c.chunk_id in st.chunk_map() for c in claims)
    assert [c.id for c in claims] == [f"C{i}" for i in range(1, len(claims) + 1)]
    # Q1 and Q2 retrieve overlapping passages; each claim must still appear once.
    assert len({c.text for c in claims}) == len(claims)


def test_is_duplicate():
    from research_agent.nodes.verify import _terms, is_duplicate

    seen = [_terms("The Eiffel Tower is 330 metres tall.")]
    assert is_duplicate("The Eiffel tower is 330 metres tall", seen)
    assert not is_duplicate("The Eiffel Tower was completed in 1889.", seen)


def test_planted_false_claim_is_flagged_not_verified(verified):
    false_claims = [c for c in verified.verified_claims if "900 metres" in c.text]
    assert false_claims, "fixture should have extracted the planted claim"
    assert all(c.status == "contradicted" for c in false_claims)
    true_claims = [c for c in verified.verified_claims if "330 metres" in c.text]
    assert true_claims and all(c.status == "verified" for c in true_claims)


def test_writer_only_sees_verified_claims(verified, deps):
    update = writer_node(verified, deps)
    writer_prompt = deps.llm.calls[-1][1]
    assert FALSE_FACT not in writer_prompt
    body, _, appendix = update["report"].partition("## Contested and unverified claims")
    assert FALSE_FACT not in body
    assert "CONTRADICTED" in appendix and "900 metres" in appendix
    assert "## References" in update["report"]


def test_evaluate_scores_report(verified, deps):
    st = advance(verified, writer_node, evaluate_node, deps=deps)
    ev = st.evaluation
    assert ev.citation_coverage > 0.5
    assert ev.cited_sources >= 2 and ev.invalid_citations == []
    assert ev.claims_contradicted >= 1 and ev.injection_flags >= 1
    assert ev.hallucination_risk == 0.0


def test_followup_adds_sources_for_uncertain_claims(verified, deps):
    uncertain = verified.verified_claims[0].model_copy(update={"status": "uncertain", "contradicting_sources": []})
    st = verified.model_copy(update={"verified_claims": [uncertain], "sources": verified.sources[:1],
                                     "chunks": [c for c in verified.chunks if c.source_id == "S1"]})
    assert needs_followup(st, max_rounds=1)
    update = followup_node(st, deps)
    assert update["verification_round"] == 1
    assert len(update["sources"]) > 1 and len(update["chunks"]) > len(st.chunks)
    assert not needs_followup(st.model_copy(update=update), max_rounds=1)


def test_reverify_skips_claims_without_new_evidence(verified, deps):
    uncertain = verified.verified_claims[0].model_copy(update={"status": "uncertain", "contradicting_sources": []})
    st = verified.model_copy(update={"verified_claims": [uncertain], "extracted_claims": [uncertain],
                                     "verification_round": 1})
    calls_before = len(deps.llm.calls)
    update = verify_node(st, deps)  # no follow-up sources were added, so nothing to re-judge
    assert len(deps.llm.calls) == calls_before
    assert update["verified_claims"][0].status == "uncertain"


def test_keyword_query_is_short():
    from research_agent.nodes.verify import keyword_query

    q = keyword_query(_claim(text="Pinecone charges $0.33 per GB of storage and $8.25 per million read "
                                  "units for serverless indexes in 2026", entities=["Pinecone"]))
    assert q.startswith("Pinecone") and len(q.split()) <= 9 and "per" not in q.split()[:2]


# ------------------------------------------------------- verification rules

def _claim(**kw):
    return Claim(**{"id": "C1", "text": "x", "source_id": "S1", "chunk_id": "S1-c0",
                    "subquestion_id": "Q1", "confidence": 0.8, **kw})


SOURCES = {
    "S1": Source(id="S1", url="u1", quality=0.55),
    "S2": Source(id="S2", url="u2", quality=0.55),
    "S3": Source(id="S3", url="u3", quality=0.9),
}


def test_decide_status_rules():
    sup = lambda s: Evidence(source_id=s, chunk_id=f"{s}-c0", stance="supports")  # noqa: E731
    con = lambda s: Evidence(source_id=s, chunk_id=f"{s}-c0", stance="contradicts")  # noqa: E731
    assert decide_status(_claim(), [sup("S1"), sup("S2")], SOURCES).status == "verified"
    assert decide_status(_claim(), [sup("S1")], SOURCES).status == "uncertain"
    assert decide_status(_claim(source_id="S3", chunk_id="S3-c0"), [sup("S3")], SOURCES).status == "verified"
    assert decide_status(_claim(), [sup("S1"), con("S2"), con("S3")], SOURCES).status == "contradicted"
    # A clear quality-weighted majority survives one dissenting source...
    disputed = decide_status(_claim(), [sup("S1"), sup("S3"), con("S2")], SOURCES)
    assert disputed.status == "verified" and "disputed by S2" in disputed.notes
    # ...a narrow one does not.
    contested = decide_status(_claim(), [sup("S1"), sup("S2"), con("S3")], SOURCES)
    assert contested.status == "uncertain" and "contested" in contested.notes
    misread = decide_status(_claim(), [Evidence(source_id="S1", chunk_id="S1-c0", stance="irrelevant")], SOURCES)
    assert misread.status == "uncertain"


def test_stale_numeric_claims_lose_confidence():
    old = {**SOURCES, "S1": SOURCES["S1"].model_copy(update={"published": "2015-01-01"})}
    sup = [Evidence(source_id=s, chunk_id=f"{s}-c0", stance="supports") for s in ("S1", "S2")]
    fresh = decide_status(_claim(numbers=["42"]), sup, SOURCES)
    stale = decide_status(_claim(numbers=["42"]), sup, old)
    assert stale.final_confidence < fresh.final_confidence and "out of date" in stale.notes


# ------------------------------------------------------------ eval helpers

def test_citation_helpers():
    text = "Growth was 40% [S1, S3]. Margins fell [S2]. Not cited here at all today."
    assert cited_source_ids(text) == ["S1", "S3", "S2"]
    assert len(statements(text)) == 1  # only sentences of 6+ words count


def test_invalid_citation_detected(verified, deps):
    st = verified.model_copy(update={"report": "# R\n\nThe tower is tall and famous worldwide [S42].\n"})
    assert evaluate(st, None).invalid_citations == ["S42"]
