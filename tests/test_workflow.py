"""The full graph and the job service: pause/resume, failure recovery, persistence."""

import time

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.db.vectors import SqlVectorStore
from research_agent.graph import make_checkpointer
from research_agent.llm import HashingEmbedder
from research_agent.nodes import Deps
from research_agent.nodes import verify as verify_mod
from research_agent.service import InvalidJobState, ResearchService
from tests.fakes import FALSE_FACT, FakeFetcher, FakeLLM, FakeSearch


@pytest.fixture
def sql_deps(settings) -> Deps:
    # SqlVectorStore exercises the real chunks table (pgvector when DATABASE_URL is Postgres).
    return Deps(llm=FakeLLM(), embedder=HashingEmbedder(settings.embedding_dim), store=SqlVectorStore(),
                search=FakeSearch(), fetcher=FakeFetcher(), settings=settings)


@pytest.fixture
def svc(settings, sql_deps) -> ResearchService:
    return ResearchService(settings, deps=sql_deps, checkpointer=InMemorySaver())


def nodes_run(svc, job_id):
    return [e["node"] for e in svc.get_job(job_id)["timeline"] if e["success"]]


def test_pauses_for_approval_then_completes(svc):
    job_id = svc.create_job("What are the key facts about the Eiffel Tower?")
    assert svc.run(job_id) == "awaiting_approval"
    job = svc.get_job(job_id)
    assert job["plan"]["subquestions"] and nodes_run(svc, job_id) == ["planner"]

    assert svc.approve(job_id, background=False) == "completed"
    job = svc.get_job(job_id)
    assert job["evaluation"]["citation_coverage"] > 0
    assert nodes_run(svc, job_id) == ["planner", "human_approval", "search", "ingest", "retrieve",
                                      "extract_claims", "verify", "writer", "evaluate"]
    report = svc.get_job_row(job_id).report
    assert FALSE_FACT not in report.split("## Contested")[0]


def test_rejected_plan_stops(svc):
    job_id = svc.create_job("What are the key facts about the Eiffel Tower?")
    svc.run(job_id)
    assert svc.approve(job_id, approved=False, background=False) == "rejected"
    assert "search" not in nodes_run(svc, job_id)


def test_approve_and_retry_mark_the_job_running_before_returning(svc, sql_deps, monkeypatch):
    """The web UI polls right after approve/retry and stops polling unless it sees "running";
    with a remote database the worker used to set that status too late."""
    submitted = []
    monkeypatch.setattr(svc, "submit", lambda job_id, resume=None: submitted.append(job_id))

    job_id = svc.create_job("What are the key facts about the Eiffel Tower?")
    assert svc.run(job_id) == "awaiting_approval"
    svc.approve(job_id)
    assert svc.get_job(job_id)["status"] == "running" and submitted == [job_id]
    with pytest.raises(InvalidJobState):  # a double click doesn't start the job twice
        svc.approve(job_id)

    failed = svc.create_job("What are the key facts about the Eiffel Tower?", auto_approve=True)
    sql_deps.llm.fail_on = verify_mod._Verdicts
    assert svc.run(failed) == "failed"
    svc.resume(failed)
    job = svc.get_job(failed)
    assert job["status"] == "running" and job["error"] is None and submitted[-1] == failed


VISITORS_Q = "How many people visit the Eiffel Tower each year?"


def completed_job(svc, auto_approve=False) -> str:
    job_id = svc.create_job("What are the key facts about the Eiffel Tower?", auto_approve=auto_approve)
    assert svc.run(job_id) == ("completed" if auto_approve else "awaiting_approval")
    if not auto_approve:
        assert svc.approve(job_id, background=False) == "completed"
    return job_id


def test_followup_edits_the_report_and_keeps_the_earlier_version(svc, sql_deps):
    from research_agent.evaluation import report_body
    from research_agent.nodes.writer import split_sections

    job_id = completed_job(svc)
    original = svc.get_job_row(job_id).report
    planner_calls = sum(1 for name, _ in sql_deps.llm.calls if name == "_PlanOut")

    assert svc.ask(job_id, VISITORS_Q, background=False) == "awaiting_approval"
    fu = svc.get_job(job_id)["followups"][-1]
    assert fu["status"] == "awaiting_approval" and fu["question"] == VISITORS_Q
    assert [q["id"] for q in fu["subquestions"]] == ["Q3"]

    assert svc.approve(job_id, background=False) == "completed"
    job, report = svc.get_job(job_id), svc.get_job_row(job_id).report
    fu = job["followups"][-1]
    assert fu["status"] == "completed" and fu["changed_sections"] == ["Visitors"] and fu["version"] == 2
    assert job["version"] == 2 and svc.report_version(job_id, 1) == original

    # Edited, not rewritten: every original section is still there word for word.
    _, before = split_sections(report_body(original))
    _, after = split_sections(report_body(report))
    assert [s for s in after if s[0] != "Visitors"] == before
    assert after[1][0] == "Visitors" and "7 million visitors" in after[1][1]

    # Only the follow-up's subquestion was researched, and the new source is cited.
    new_source = next(s for s in svc.state(job_id).sources if "visitors" in s.url)
    assert f"[{new_source.id}]" in report.split("## References")[1]
    assert sum(1 for name, _ in sql_deps.llm.calls if name == "_PlanOut") == planner_calls
    claims = svc.state(job_id).verified_claims
    assert any(c.subquestion_id == "Q3" and c.status == "verified" for c in claims)
    assert FALSE_FACT not in report.split("## Contested")[0]


def test_followup_edit_that_drops_content_keeps_the_section_and_the_rest(svc, sql_deps):
    """A rewrite that loses a section's citations is redone; if it still loses them, that section
    stays as it was, and the rest of the edit (here, a new section) is kept."""
    from research_agent.nodes import writer as writer_mod

    careless = writer_mod._Edit(action="replace", section="Executive summary", content="A shorter summary.")

    class CarelessLLM(type(sql_deps.llm)):
        def structured(self, system, user, schema, max_tokens=4000):
            if schema is writer_mod._Revision:
                good = super().structured(system, user, schema, max_tokens).edits
                first = sum(n == "_Revision" for n, _ in self.calls) == 1
                return schema(edits=[careless, *good] if first else [careless], summary="Shortened it.")
            return super().structured(system, user, schema, max_tokens)

    job_id = completed_job(svc, auto_approve=True)
    summary_before = svc.get_job_row(job_id).report.split("## Conclusion")[0].split("## Executive summary")[1]
    sql_deps.llm.__class__ = CarelessLLM
    assert svc.ask(job_id, VISITORS_Q, background=False) == "completed"
    prompts = [u for n, u in sql_deps.llm.calls if n == "_Revision"]
    assert len(prompts) == 2 and '"Executive summary" no longer cites' in prompts[1]

    report = svc.get_job_row(job_id).report
    assert summary_before.strip() in report and "A shorter summary." not in report
    assert "## Visitors" in report
    fu = svc.get_job(job_id)["followups"][-1]
    assert fu["changed_sections"] == ["Visitors"] and fu["summary"] == "Updated: Visitors." and fu["version"] == 2


def test_edited_followup_plan_is_what_gets_researched(svc, sql_deps):
    job_id = completed_job(svc)
    svc.ask(job_id, VISITORS_Q, background=False)
    edited = {"subquestions": [{"id": "Q3", "question": "How many visitors does the Eiffel Tower get a year?",
                                "search_queries": []}]}
    assert svc.approve(job_id, plan=edited, background=False) == "completed"
    subqs = svc.state(job_id).plan.subquestions
    assert [(q.id, q.question) for q in subqs][-1] == ("Q3", "How many visitors does the Eiffel Tower get a year?")
    assert len(subqs) == 3 and "How many visitors does the Eiffel Tower get a year?" in sql_deps.search.calls


def test_rejected_followup_leaves_the_report_alone(svc):
    job_id = completed_job(svc)
    original = svc.get_job_row(job_id).report
    assert svc.ask(job_id, VISITORS_Q, background=False) == "awaiting_approval"
    assert svc.approve(job_id, approved=False, background=False) == "completed"
    job = svc.get_job(job_id)
    assert job["followups"][-1]["status"] == "rejected" and job["version"] == 1
    assert svc.get_job_row(job_id).report == original
    assert [q.id for q in svc.state(job_id).plan.subquestions] == ["Q1", "Q2"]


def test_followups_on_auto_approved_jobs_run_straight_through(svc):
    job_id = completed_job(svc, auto_approve=True)
    assert svc.ask(job_id, VISITORS_Q, background=False) == "completed"
    assert svc.ask(job_id, "Which other monuments in Paris get many visitors?", background=False) == "completed"
    job = svc.get_job(job_id)
    assert [f["status"] for f in job["followups"]] == ["completed", "completed"]
    # The second follow-up found nothing new to verify, so the text stayed the same.
    assert job["followups"][1]["version"] is None and job["version"] == 2


def test_followup_needs_a_completed_report(svc):
    job_id = svc.create_job("What are the key facts about the Eiffel Tower?")
    svc.run(job_id)
    with pytest.raises(InvalidJobState):
        svc.ask(job_id, VISITORS_Q)


def test_apply_edits_only_touches_the_named_sections():
    from research_agent.nodes.writer import _Edit, apply_edits

    body = "# T\n\n## Executive summary\n\nA [S1].\n\n## Height\n\nB [S2].\n\n## Conclusion\n\nC."
    out, changed = apply_edits(body, [
        _Edit(action="replace", section="height", content="B, now 330 m [S2][S4]."),
        _Edit(action="insert", section="Not a heading", heading="Visitors", content="D [S4]."),
    ])
    assert out == ("# T\n\n## Executive summary\n\nA [S1].\n\n## Height\n\nB, now 330 m [S2][S4]."
                   "\n\n## Visitors\n\nD [S4].\n\n## Conclusion\n\nC.")
    assert changed == ["Height", "Visitors"]

    out, changed = apply_edits(body, [_Edit(action="append", section="Conclusion", content="E [S4]."),
                                      _Edit(action="replace", section="Height", content="B [S2].")])
    assert out.endswith("## Conclusion\n\nC.\n\nE [S4].") and changed == ["Conclusion"]  # a no-op replace


def test_failed_node_resumes_without_rerunning_earlier_nodes(svc, sql_deps):
    job_id = svc.create_job("What are the key facts about the Eiffel Tower?", auto_approve=True)
    sql_deps.llm.fail_on = verify_mod._Verdicts
    assert svc.run(job_id) == "failed"
    assert "simulated failure" in svc.get_job(job_id)["error"]
    planner_calls = sum(1 for name, _ in sql_deps.llm.calls if name == "_PlanOut")

    sql_deps.llm.fail_on = None
    assert svc.resume(job_id, background=False) == "completed"
    assert sum(1 for name, _ in sql_deps.llm.calls if name == "_PlanOut") == planner_calls
    ok = nodes_run(svc, job_id)
    assert ok.count("planner") == 1 and ok.count("search") == 1 and ok[-1] == "evaluate"


def test_job_state_survives_restart(settings, sql_deps, tmp_path):
    """A new service (fresh process) picks up a paused job from the persistent checkpointer."""
    cp_settings = settings.model_copy(update={"checkpoint_url": f"sqlite:///{tmp_path.as_posix()}/cp.db"})
    first = ResearchService(cp_settings, deps=sql_deps, checkpointer=make_checkpointer(cp_settings))
    job_id = first.create_job("What are the key facts about the Eiffel Tower?")
    assert first.run(job_id) == "awaiting_approval"
    del first

    second = ResearchService(cp_settings, deps=sql_deps, checkpointer=make_checkpointer(cp_settings))
    assert second.state(job_id).plan is not None
    assert second.approve(job_id, background=False) == "completed"


def test_init_db_creates_missing_data_folder(tmp_path):
    """A fresh clone has no data/ folder (it's git-ignored); startup must create it."""
    import sqlalchemy as sa

    from research_agent.db.session import init_db

    db = tmp_path / "fresh" / "data" / "research.db"
    init_db(f"sqlite:///{db.as_posix()}")
    assert "research_jobs" in sa.inspect(sa.create_engine(f"sqlite:///{db.as_posix()}")).get_table_names()


def test_hosted_postgres_urls_get_the_psycopg_driver(settings):
    from research_agent.config import Settings

    neon = Settings(_env_file=None, database_url="postgresql://u:p@ep-x-pooler.neon.tech/db?sslmode=require",
                    database_url_unpooled="postgres://u:p@ep-x.neon.tech/db?sslmode=require")
    assert neon.database_url.startswith("postgresql+psycopg://u:p@ep-x-pooler.")
    assert neon.migration_url == "postgresql+psycopg://u:p@ep-x.neon.tech/db?sslmode=require"
    local = Settings(_env_file=None, database_url="sqlite:///./x.db", database_url_unpooled=None)
    assert local.migration_url == "sqlite:///./x.db"


class FakeStore:
    def __init__(self, fail=False):
        self.objects, self.fail = {}, fail

    def put(self, key, body, content_type):
        if self.fail:
            raise ConnectionError("bucket unreachable")
        self.objects[key] = (body, content_type)

    def url(self, key, expires_s=3600):
        return f"https://storage.test/{key}?signed"


def test_completed_job_uploads_artifacts_and_links_them(svc):
    svc.artifacts = FakeStore()
    job_id = svc.create_job("What are the key facts about the Eiffel Tower?", auto_approve=True)
    assert svc.run(job_id) == "completed"
    assert set(svc.artifacts.objects) == {f"reports/{job_id}/report.md", f"reports/{job_id}/run.json"}
    body, ctype = svc.artifacts.objects[f"reports/{job_id}/report.md"]
    assert body.decode() == svc.get_job_row(job_id).report and ctype.startswith("text/markdown")
    assert svc.get_job(job_id)["artifacts"]["report.md"].startswith(f"https://storage.test/reports/{job_id}/")


def test_storage_outage_does_not_fail_the_job(svc):
    svc.artifacts = FakeStore(fail=True)
    job_id = svc.create_job("What are the key facts about the Eiffel Tower?", auto_approve=True)
    assert svc.run(job_id) == "completed"
    assert svc.get_job(job_id)["artifacts"] == {} and svc.get_job_row(job_id).report


def test_spent_today_counts_cost_and_reserves_active_jobs(svc, settings):
    from research_agent.db import ResearchJob, session_scope

    baseline = svc.spent_today()
    done, active = svc.create_job("finished question here"), svc.create_job("running question here")
    with session_scope() as s:
        s.get(ResearchJob, done).cost_usd, s.get(ResearchJob, done).status = 0.25, "completed"
        s.get(ResearchJob, active).status = "running"
    assert svc.spent_today() == pytest.approx(baseline + 0.25 + settings.max_job_cost_usd)
    with session_scope() as s:
        s.get(ResearchJob, active).status = "failed"


def test_budget_cap_fails_job(settings, sql_deps):
    class PricedLLM(FakeLLM):
        model = "gpt-4.1-mini"

        def structured(self, *a, **kw):
            from research_agent.llm import _record
            _record(self.model, 200_000, 0)  # $0.08 per call
            return super().structured(*a, **kw)

    sql_deps.llm = PricedLLM()
    svc = ResearchService(settings.model_copy(update={"max_job_cost_usd": 0.10}), deps=sql_deps,
                          checkpointer=InMemorySaver())
    job_id = svc.create_job("What are the key facts about the Eiffel Tower?", auto_approve=True)
    assert svc.run(job_id) == "failed"
    job = svc.get_job(job_id)
    assert "exceeded budget" in job["error"] and job["cost_usd"] > 0.10


# ----------------------------------------------------------------------- API

@pytest.fixture
def client(svc):
    from research_agent.api.main import app, limiter

    limiter.reset()
    app.state.service = svc
    with TestClient(app) as c:
        yield c
    app.state.service = None


def wait_for(client, job_id, statuses, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/research/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job stuck in {job['status']}")


def test_api_end_to_end(client):
    r = client.post("/research", json={"question": "What are the key facts about the Eiffel Tower?"})
    assert r.status_code == 202 and "x-request-id" in r.headers
    job_id = r.json()["job_id"]

    job = wait_for(client, job_id, {"awaiting_approval"})
    assert client.get(f"/research/{job_id}/report").status_code == 409

    plan = job["plan"]
    plan["subquestions"] = plan["subquestions"][:1]
    assert client.post(f"/research/{job_id}/approve", json={"approved": True, "plan": plan}).status_code == 202
    job = wait_for(client, job_id, {"completed", "failed"})
    assert job["status"] == "completed", job["error"]
    assert len(job["plan"]["subquestions"]) == 1

    md = client.get(f"/research/{job_id}/report")
    assert md.headers["content-type"].startswith("text/markdown") and "## References" in md.text
    full = client.get(f"/research/{job_id}/report?format=json").json()
    assert full["sources"] and full["claims"] and full["evaluation"]
    assert any(j["job_id"] == job_id for j in client.get("/research").json())


def test_api_followup(client):
    job_id = client.post("/research", json={"question": "What are the key facts about the Eiffel Tower?",
                                            "auto_approve": True}).json()["job_id"]
    assert client.post(f"/research/{job_id}/followup", json={"question": VISITORS_Q}).status_code == 409
    wait_for(client, job_id, {"completed"})
    original = client.get(f"/research/{job_id}/report").text

    r = client.post(f"/research/{job_id}/followup", json={"question": VISITORS_Q})
    assert r.status_code == 202
    # Listed and running at once, so the UI can follow it.
    job = client.get(f"/research/{job_id}").json()
    assert job["followups"][-1]["question"] == VISITORS_Q and job["status"] in ("running", "completed")
    job = wait_for(client, job_id, {"completed", "failed"})
    assert job["status"] == "completed" and job["version"] == 2, job["error"]
    assert "## Visitors" in client.get(f"/research/{job_id}/report").text
    assert client.get(f"/research/{job_id}/report?version=1").text == original
    assert client.get(f"/research/{job_id}/report?version=9").status_code == 404
    assert client.post(f"/research/{job_id}/followup", json={"question": "short"}).status_code == 422


def test_api_validation_and_404(client):
    assert client.post("/research", json={"question": "short"}).status_code == 422
    assert client.get("/research/nope").status_code == 404
    job_id = client.post("/research", json={"question": "What are the key facts about the Eiffel Tower?",
                                            "auto_approve": True}).json()["job_id"]
    wait_for(client, job_id, {"completed"})
    assert client.post(f"/research/{job_id}/approve", json={"approved": True}).status_code == 409


def test_api_models_and_model_selection(client, svc):
    models = client.get("/models").json()
    assert models == [{"id": "fake-model", "label": "fake-model", "provider": "custom", "default": True}]

    bad = client.post("/research", json={"question": "What are the key facts about the Eiffel Tower?",
                                         "model": "openai/not-a-model"})
    assert bad.status_code == 422 and "unknown model" in bad.json()["detail"]

    job_id = client.post("/research", json={"question": "What are the key facts about the Eiffel Tower?",
                                            "model": "fake-model", "auto_approve": True}).json()["job_id"]
    job = wait_for(client, job_id, {"completed"})
    assert job["model"] == "fake-model" and job["model_label"] == "fake-model"
    assert client.get("/").status_code == 200


def test_available_models_follow_configured_keys(settings):
    from research_agent.llm import available_models

    openai_only = settings.model_copy(update={"openai_api_key": "k", "anthropic_api_key": None, "llm_provider": None})
    assert [m.id for m in available_models(openai_only)] == ["openai/gpt-4.1-mini"]

    both = openai_only.model_copy(update={"anthropic_api_key": "k", "llm_provider": "anthropic"})
    assert [m.id for m in available_models(both)] == ["anthropic/claude-haiku-5-5", "openai/gpt-4.1-mini"]

    custom = openai_only.model_copy(update={"openai_model": "gpt-4.1"})
    assert [m.id for m in available_models(custom)][0] == "openai/gpt-4.1"


def test_api_rate_limit(client, svc, monkeypatch):
    monkeypatch.setattr(svc, "spent_today", lambda: 0.0)  # isolate from the cost cap's reservations
    codes = [client.post("/research", json={"question": "What are the key facts about the Eiffel Tower?",
                                            "auto_approve": True}).status_code for _ in range(6)]
    assert codes[:5] == [202] * 5 and codes[5] == 429


def test_api_daily_cost_cap(client, svc, monkeypatch):
    monkeypatch.setattr(svc, "spent_today", lambda: 999.0)
    r = client.post("/research", json={"question": "What are the key facts about the Eiffel Tower?"})
    assert r.status_code == 429 and "cap" in r.json()["detail"]
