"""Job lifecycle on top of the graph: create, run, pause for approval, resume, recover.

The research_jobs row is the public view of a job; the LangGraph checkpoint is the
source of truth for its workflow state. `_sync` copies the latter into the former
after every run segment.
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import structlog
from langgraph.types import Command
from sqlalchemy import func, select, update

from research_agent.config import Settings, get_settings
from research_agent.db import ResearchJob, TimelineRecord, session_scope
from research_agent.db.vectors import SqlVectorStore
from research_agent.evaluation import finalize_totals
from research_agent.graph import build_graph, make_checkpointer
from research_agent.llm import (
    CATALOG,
    BudgetExceeded,
    ModelOption,
    available_models,
    make_embedder,
    make_llm,
    metered,
)
from research_agent.logging import get_logger
from research_agent.nodes import Deps
from research_agent.state import FollowUp, ResearchState, TimelineEvent
from research_agent.storage import make_artifact_store
from research_agent.tools.fetch import PageFetcher
from research_agent.tools.search import make_search

log = get_logger(__name__)

ACTIVE = ("queued", "running")


class JobNotFound(KeyError):
    pass


class InvalidJobState(RuntimeError):
    pass


def default_deps(settings: Settings) -> Deps:
    return Deps(
        llm=make_llm(settings),
        embedder=make_embedder(settings),
        store=SqlVectorStore(),
        search=make_search(settings),
        fetcher=PageFetcher(settings.fetch_timeout_s, settings.max_doc_chars),
        settings=settings,
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _start_of_day() -> datetime:
    return _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


class ResearchService:
    def __init__(self, settings: Settings | None = None, deps: Deps | None = None, checkpointer=None):
        self.settings = settings or get_settings()
        if deps is None:
            self.models = available_models(self.settings)
            deps = default_deps(self.settings)
        else:  # injected deps (tests): their LLM is the only model on offer
            self.models = [ModelOption(deps.llm.model, "custom", deps.llm.model, deps.llm.model)]
        self.deps = deps
        self.deps.on_event = self._record_event
        self.checkpointer = checkpointer or make_checkpointer(self.settings)
        self.artifacts = make_artifact_store(self.settings)
        # One compiled graph per model, all sharing the checkpointer; the default model's
        # graph also serves state reads, since every graph has the same shape.
        self.default_model = self.models[0].id if self.models else self.deps.llm.model
        self.graph = build_graph(self.deps, self.checkpointer)
        self._graphs = {self.default_model: self.graph}
        self.pool = ThreadPoolExecutor(max_workers=self.settings.worker_threads, thread_name_prefix="job")
        self._lock = threading.Lock()
        self._running: set[str] = set()

    def graph_for(self, model_id: str | None):
        option = next((m for m in self.models if m.id == model_id), None)
        if option is None:  # e.g. a job from before model selection existed
            return self.graph
        with self._lock:
            if option.id not in self._graphs:
                deps = replace(self.deps, llm=make_llm(self.settings, option))
                self._graphs[option.id] = build_graph(deps, self.checkpointer)
            return self._graphs[option.id]

    # ------------------------------------------------------------ public API

    def create_job(self, question: str, auto_approve: bool = False, client_ip: str | None = None,
                   model: str | None = None) -> str:
        model = model or self.default_model
        if model not in {m.id for m in self.models}:
            raise ValueError(f"unknown model {model!r}")
        job_id = str(uuid.uuid4())
        with session_scope() as s:
            s.add(ResearchJob(id=job_id, question=question.strip(), status="queued",
                              model=model, auto_approve=auto_approve, client_ip=client_ip))
        log.info("job_created", job_id=job_id, model=model)
        return job_id

    def submit(self, job_id: str, resume: dict | None = None, followup: FollowUp | None = None) -> Future:
        """Runs the next segment of a job in the background."""
        return self.pool.submit(self.run, job_id, resume, followup)

    def run(self, job_id: str, resume: dict | None = None, followup: FollowUp | None = None) -> str:
        """Runs a job until it completes, fails, or pauses for approval. Returns the new status.

        `followup` starts a reader's follow-up question on a completed job."""
        with self._lock:
            if job_id in self._running:
                raise InvalidJobState("job is already running")
            self._running.add(job_id)
        structlog.contextvars.bind_contextvars(job_id=job_id)
        try:
            return self._run(job_id, resume, followup)
        finally:
            structlog.contextvars.unbind_contextvars("job_id")
            with self._lock:
                self._running.discard(job_id)

    def approve(self, job_id: str, approved: bool = True, plan: dict | None = None, background: bool = True):
        if not self._claim(job_id, ("awaiting_approval",)):
            raise InvalidJobState(f"job is {self.get_job_row(job_id).status}, not awaiting approval")
        decision = {"approved": approved, "plan": plan}
        return self.submit(job_id, decision) if background else self.run(job_id, decision)

    def ask(self, job_id: str, question: str, background: bool = True):
        """Researches a follow-up question on a completed job and edits its report."""
        if not self._claim(job_id, ("completed",)):
            raise InvalidJobState(f"job is {self.get_job_row(job_id).status}; follow-up questions "
                                  "can be asked once the report is complete")
        job = self.get_job_row(job_id)
        fu = FollowUp(question=question.strip(), asked_at=_utcnow(), base_cost_usd=job.cost_usd or 0.0)
        # Listed straight away, so a client polling the job can show the follow-up's progress.
        result = dict(job.result or {})
        result["followups"] = result.get("followups", []) + [
            fu.model_dump(mode="json", exclude={"base_cost_usd"}) | {"subquestions": []}]
        self._update(job_id, result=result)
        return self.submit(job_id, followup=fu) if background else self.run(job_id, followup=fu)

    def resume(self, job_id: str, background: bool = True):
        """Retries a failed job from its last checkpoint (completed nodes are not re-run)."""
        if not self._claim(job_id, ("failed", "queued", "running")):
            raise InvalidJobState(f"job is {self.get_job_row(job_id).status}; "
                                  "only failed or interrupted jobs can be resumed")
        return self.submit(job_id) if background else self.run(job_id)

    def recover_inflight(self) -> list[str]:
        """Called at startup: resumes jobs a previous process left mid-run."""
        with session_scope() as s:
            ids = list(s.scalars(select(ResearchJob.id).where(ResearchJob.status.in_(ACTIVE))))
        for job_id in ids:
            log.info("recovering_job", job_id=job_id)
            self.submit(job_id)
        return ids

    def spent_today(self) -> float:
        with session_scope() as s:
            spent = s.scalar(
                select(func.coalesce(func.sum(ResearchJob.cost_usd), 0.0))
                .where(ResearchJob.created_at >= _start_of_day())
            )
            active = s.scalar(select(func.count()).where(ResearchJob.status.in_(ACTIVE)))
        # Reserve the per-job budget for jobs still running.
        return float(spent) + active * self.settings.max_job_cost_usd

    def get_job_row(self, job_id: str) -> ResearchJob:
        with session_scope() as s:
            job = s.get(ResearchJob, job_id)
            if job is None:
                raise JobNotFound(job_id)
            return job

    def get_job(self, job_id: str) -> dict:
        job = self.get_job_row(job_id)
        with session_scope() as s:
            events = s.scalars(
                select(TimelineRecord).where(TimelineRecord.job_id == job_id).order_by(TimelineRecord.id)
            ).all()
        return {
            "job_id": job.id,
            "question": job.question,
            "status": job.status,
            "objective": job.objective,
            "plan": job.plan,
            "model": job.model,
            "model_label": self.model_label(job.model),
            "cost_usd": round(job.cost_usd or 0.0, 5),
            "error": job.error,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "evaluation": job.evaluation,
            "timeline": [
                {"node": e.node, "started_at": e.started_at, "duration_ms": e.duration_ms,
                 "success": e.success, "error": e.error, "cost_usd": round(e.cost_usd, 5)}
                for e in events
            ],
            "has_report": job.report is not None,
            "artifacts": self.artifact_links(job),
            "followups": (job.result or {}).get("followups", []),
            "version": len((job.result or {}).get("versions", [])) + 1,
        }

    def report_version(self, job_id: str, version: int) -> str | None:
        """An earlier version of the report (1 = the original); the current one is the last."""
        job = self.get_job_row(job_id)
        versions = (job.result or {}).get("versions", []) + [job.report]
        return versions[version - 1] if 1 <= version <= len(versions) else None

    def list_jobs(self, limit: int = 20) -> list[dict]:
        with session_scope() as s:
            rows = s.scalars(select(ResearchJob).order_by(ResearchJob.created_at.desc()).limit(limit)).all()
            return [{"job_id": r.id, "question": r.question, "status": r.status,
                     "model_label": self.model_label(r.model), "created_at": r.created_at,
                     "cost_usd": round(r.cost_usd or 0.0, 5)} for r in rows]

    def model_label(self, model_id: str | None) -> str:
        known = {m.id: m.label for m in [*self.models, *CATALOG]}
        return known.get(model_id or "", model_id or "")

    def state(self, job_id: str) -> ResearchState | None:
        snap = self.graph.get_state({"configurable": {"thread_id": job_id}})
        return ResearchState.model_validate(snap.values) if snap.values else None

    # ------------------------------------------------------------ internals

    def _run(self, job_id: str, resume: dict | None, followup: FollowUp | None = None) -> str:
        job = self.get_job_row(job_id)
        config = {"configurable": {"thread_id": job_id}}
        snap = self.graph.get_state(config)
        st = ResearchState.model_validate(snap.values) if snap.values else None
        if followup is not None:
            # Re-enter the finished thread from START; it routes to the follow-up planner.
            st = st.model_copy(update={"followups": st.followups + [followup], "active": [],
                                       "verification_round": 0})
            graph_input = st.model_dump()
        elif resume is not None:
            graph_input = Command(resume=resume)
        elif snap.values:
            graph_input = None  # continue from the last checkpoint
        else:
            graph_input = ResearchState(job_id=job_id, question=job.question,
                                        auto_approve=job.auto_approve).model_dump()

        self._update(job_id, status="running", error=None)
        try:
            pending = st.pending_followup() if st else None
            budget = self.settings.max_job_cost_usd + (pending.base_cost_usd if pending else 0.0)
            with metered(budget_usd=budget, base_cost_usd=job.cost_usd or 0.0):
                self.graph_for(job.model).invoke(graph_input, config)
        except BudgetExceeded as e:
            log.error("job_budget_exceeded", error=str(e))
            return self._sync(job_id, error=str(e))
        except Exception as e:
            log.exception("job_failed")
            return self._sync(job_id, error=f"{type(e).__name__}: {e}"[:1000])
        return self._sync(job_id)

    def _sync(self, job_id: str, error: str | None = None) -> str:
        snap = self.graph.get_state({"configurable": {"thread_id": job_id}})
        st = ResearchState.model_validate(snap.values) if snap.values else None
        fields: dict = {}
        if st and st.plan:
            fields.update(objective=st.plan.objective, plan=st.plan.model_dump())
        result = dict(self.get_job_row(job_id).result or {})
        if st and st.followups:
            result.update(self._followup_summary(st))
            fields["result"] = result

        if error:
            status = "failed"
            fields["error"] = error
        elif snap.interrupts:
            status = "awaiting_approval"
        elif not snap.next and st and st.approved is False:
            status = "rejected"
            fields["completed_at"] = _utcnow()
        elif not snap.next and st and st.report:
            status = "completed"
            evaluation = finalize_totals(st.evaluation, st) if st.evaluation else None
            result = self._result_summary(st) | self._followup_summary(st)
            result["artifacts"] = self._save_artifacts(st, evaluation)
            fields.update(
                report=st.report,
                evaluation=evaluation.model_dump() if evaluation else None,
                result=result,
                completed_at=_utcnow(),
            )
        else:
            status = "failed"
            fields["error"] = "workflow stopped unexpectedly"
        self._update(job_id, status=status, **fields)
        log.info("job_status", status=status)
        return status

    @staticmethod
    def _result_summary(st: ResearchState) -> dict:
        return {
            "sources": [s.model_dump(include={"id", "url", "title", "domain", "published", "fetched",
                                              "quality", "injection_flags"}) for s in st.sources],
            "claims": [c.model_dump(include={"id", "text", "status", "source_id", "supporting_sources",
                                             "contradicting_sources", "notes", "final_confidence"})
                       for c in st.verified_claims],
            "errors": st.errors,
        }

    @staticmethod
    def _followup_summary(st: ResearchState) -> dict:
        subqs = {sq.id: sq for sq in st.plan.subquestions} if st.plan else {}
        followups = []
        for fu in st.followups:
            item = fu.model_dump(mode="json", exclude={"base_cost_usd"})
            item["subquestions"] = [subqs[i].model_dump() for i in fu.subquestion_ids if i in subqs]
            followups.append(item)
        return {"followups": followups, "versions": st.report_versions}

    def _save_artifacts(self, st: ResearchState, evaluation) -> dict[str, str]:
        """Write the report and full run record locally, and to object storage when configured.

        Returns the object keys that were uploaded ({} without a store). An upload failure
        is logged, not raised: the report is already safe in the database.
        """
        files = {
            "report.md": ((st.report or "").encode("utf-8"), "text/markdown; charset=utf-8"),
            "run.json": (json.dumps({"state": st.model_dump(mode="json", exclude={"chunks"}),
                                     "evaluation": evaluation.model_dump() if evaluation else None},
                                    indent=2).encode("utf-8"), "application/json"),
        }
        for n, text in enumerate(st.report_versions, 1):  # earlier versions, after follow-ups
            files[f"report.v{n}.md"] = (text.encode("utf-8"), "text/markdown; charset=utf-8")
        out = Path(self.settings.reports_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, (body, _) in files.items():  # report.md -> <job>.md, report.v1.md -> <job>.v1.md
            (out / f"{st.job_id}{name.removeprefix('report').removeprefix('run')}").write_bytes(body)

        keys: dict[str, str] = {}
        if self.artifacts is not None:
            for name, (body, content_type) in files.items():
                key = f"reports/{st.job_id}/{name}"
                try:
                    self.artifacts.put(key, body, content_type)
                    keys[name] = key
                except Exception:
                    log.exception("artifact_upload_failed", key=key)
        return keys

    def artifact_links(self, job: ResearchJob) -> dict[str, str]:
        keys = (job.result or {}).get("artifacts") or {}
        if self.artifacts is None:
            return {}
        return {name: self.artifacts.url(key) for name, key in keys.items()}

    def _update(self, job_id: str, **fields) -> None:
        with session_scope() as s:
            job = s.get(ResearchJob, job_id)
            for k, v in fields.items():
                setattr(job, k, v)

    def _claim(self, job_id: str, from_statuses: tuple[str, ...]) -> bool:
        """Marks a job running before the worker picks it up, so a client polling right after
        approve/resume sees "running" rather than the old status. Atomic: of two concurrent
        approvals, only one wins."""
        self.get_job_row(job_id)  # JobNotFound for unknown ids
        with session_scope() as s:
            claimed = s.execute(
                update(ResearchJob)
                .where(ResearchJob.id == job_id, ResearchJob.status.in_(from_statuses))
                .values(status="running", error=None)
            ).rowcount
        return claimed == 1

    def _record_event(self, job_id: str, event: TimelineEvent) -> None:
        with session_scope() as s:
            s.add(TimelineRecord(job_id=job_id, **event.model_dump()))
            job = s.get(ResearchJob, job_id)
            if job is not None:
                job.cost_usd = (job.cost_usd or 0.0) + event.cost_usd
