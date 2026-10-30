"""HTTP API.

    GET  /models                   models the server can use
    POST /research                 submit a question            -> 202 {job_id, status}
    GET  /research                 recent jobs
    GET  /research/{id}            status, plan, timeline, metrics
    POST /research/{id}/approve    approve / edit / reject the plan
    POST /research/{id}/resume     retry a failed job from its last checkpoint
    GET  /research/{id}/report     the report (Markdown, or JSON with sources & claims)
    GET  /health
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from research_agent import __version__
from research_agent.config import get_settings
from research_agent.db import init_db
from research_agent.logging import configure_logging, get_logger
from research_agent.service import InvalidJobState, JobNotFound, ResearchService

log = get_logger(__name__)
settings = get_settings()
limiter = Limiter(key_func=get_remote_address)
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level, settings.log_json)
    init_db()
    if getattr(app.state, "service", None) is None:  # tests inject one with fake deps
        app.state.service = ResearchService(settings)
    recovered = app.state.service.recover_inflight()
    log.info("api_started", version=__version__, provider=settings.provider, model=settings.model,
             recovered_jobs=len(recovered))
    yield
    app.state.service.pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Autonomous Research Agent", version=__version__, lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limited(request: Request, exc: RateLimitExceeded):
    return JSONResponse({"detail": f"rate limit exceeded ({exc.detail}); try again shortly"}, status_code=429)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled_error", path=request.url.path)
        response = JSONResponse({"detail": "internal error", "request_id": request_id}, status_code=500)
    response.headers["x-request-id"] = request_id
    if request.url.path != "/health":
        log.info("http_request", method=request.method, path=request.url.path, status=response.status_code)
    return response


def service(request: Request) -> ResearchService:
    return request.app.state.service


class ResearchRequest(BaseModel):
    question: str = Field(min_length=10, max_length=500)
    auto_approve: bool = False
    model: str | None = Field(default=None, description="An id from GET /models; omit for the default")


class ApprovalRequest(BaseModel):
    approved: bool = True
    plan: dict | None = Field(default=None, description="Optionally, an edited plan to use instead")


def _job_or_404(svc: ResearchService, job_id: str) -> dict:
    try:
        return svc.get_job(job_id)
    except JobNotFound:
        raise HTTPException(404, "job not found")


@app.post("/research", status_code=202)
@limiter.limit(settings.rate_limit)
def submit_research(request: Request, body: ResearchRequest):
    svc = service(request)
    if body.model is not None and body.model not in {m.id for m in svc.models}:
        raise HTTPException(422, f"unknown model {body.model!r}; see GET /models")
    if svc.spent_today() >= settings.daily_cost_cap_usd:
        raise HTTPException(429, "the demo's daily spend cap has been reached; please try again tomorrow")
    job_id = svc.create_job(body.question, body.auto_approve, get_remote_address(request), body.model)
    svc.submit(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/models")
def list_models(request: Request):
    svc = service(request)
    return [{"id": m.id, "label": m.label, "provider": m.provider, "default": m.id == svc.default_model}
            for m in svc.models]


@app.get("/research")
def list_research(request: Request, limit: int = 20):
    return service(request).list_jobs(min(limit, 100))


@app.get("/research/{job_id}")
def get_research(request: Request, job_id: str):
    return _job_or_404(service(request), job_id)


@app.post("/research/{job_id}/approve", status_code=202)
@limiter.limit("20/minute")
def approve_research(request: Request, job_id: str, body: ApprovalRequest):
    svc = service(request)
    _job_or_404(svc, job_id)
    try:
        svc.approve(job_id, body.approved, body.plan)
    except InvalidJobState as e:
        raise HTTPException(409, str(e))
    return {"job_id": job_id, "status": "running" if body.approved else "rejecting"}


@app.post("/research/{job_id}/resume", status_code=202)
@limiter.limit("10/minute")
def resume_research(request: Request, job_id: str):
    svc = service(request)
    _job_or_404(svc, job_id)
    if svc.spent_today() >= settings.daily_cost_cap_usd:
        raise HTTPException(429, "the demo's daily spend cap has been reached")
    try:
        svc.resume(job_id)
    except InvalidJobState as e:
        raise HTTPException(409, str(e))
    return {"job_id": job_id, "status": "running"}


@app.get("/research/{job_id}/report")
def get_report(request: Request, job_id: str, format: str = "md"):
    svc = service(request)
    _job_or_404(svc, job_id)
    job = svc.get_job_row(job_id)
    if job.report is None:
        raise HTTPException(409, f"report not ready (status: {job.status})")
    if format == "json":
        return {"job_id": job_id, "question": job.question, "report": job.report,
                "evaluation": job.evaluation, **(job.result or {})}
    return PlainTextResponse(job.report, media_type="text/markdown; charset=utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
