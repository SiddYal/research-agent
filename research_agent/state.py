"""The single ResearchState object passed between every workflow node.

Nodes return partial updates (a dict of changed fields); LangGraph merges them and
checkpoints the result after each node, which is what makes a job resumable and
each node testable against a fixed state fixture.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ClaimStatus = Literal["verified", "uncertain", "contradicted"]


class SubQuestion(BaseModel):
    id: str
    question: str
    search_queries: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    objective: str
    scope: str = ""
    outline: list[str] = Field(default_factory=list)
    subquestions: list[SubQuestion] = Field(default_factory=list)


class Source(BaseModel):
    id: str  # "S1", "S2", ... – the citation key used in the report
    url: str
    title: str = ""
    snippet: str = ""
    query: str = ""
    subquestion_id: str = ""
    domain: str = ""
    published: str | None = None
    fetched: bool = False
    content: str = ""
    quality: float = 0.5
    injection_flags: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str  # "<source_id>-c<n>"
    source_id: str
    text: str
    section: str = ""
    position: int = 0
    flagged: bool = False


class Claim(BaseModel):
    id: str
    text: str
    source_id: str
    chunk_id: str
    subquestion_id: str
    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class Evidence(BaseModel):
    source_id: str
    chunk_id: str
    stance: Literal["supports", "contradicts", "irrelevant"]
    note: str = ""


class VerifiedClaim(Claim):
    status: ClaimStatus = "uncertain"
    supporting_sources: list[str] = Field(default_factory=list)
    contradicting_sources: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    notes: str = ""
    freshness: str | None = None
    final_confidence: float = 0.0


class TimelineEvent(BaseModel):
    node: str
    started_at: datetime
    ended_at: datetime
    duration_ms: int
    success: bool
    error: str | None = None
    cost_usd: float = 0.0
    llm_calls: int = 0
    tokens: int = 0


class Evaluation(BaseModel):
    citation_coverage: float = 0.0
    cited_sources: int = 0
    unique_domains: int = 0
    source_diversity: float = 0.0
    invalid_citations: list[str] = Field(default_factory=list)
    hallucination_risk: float = 0.0
    unsupported_statements: list[str] = Field(default_factory=list)
    claims_total: int = 0
    claims_verified: int = 0
    claims_uncertain: int = 0
    claims_contradicted: int = 0
    injection_flags: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    llm_calls: int = 0


class ResearchState(BaseModel):
    job_id: str
    question: str
    auto_approve: bool = False
    approved: bool | None = None
    plan: Plan | None = None
    sources: list[Source] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    retrieved: dict[str, list[str]] = Field(default_factory=dict)  # subquestion id -> chunk ids
    extracted_claims: list[Claim] = Field(default_factory=list)
    verified_claims: list[VerifiedClaim] = Field(default_factory=list)
    verification_round: int = 0
    report: str | None = None
    evaluation: Evaluation | None = None
    timeline: list[TimelineEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0

    def source(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def chunk_map(self) -> dict[str, Chunk]:
        return {c.id: c for c in self.chunks}

    def next_source_id(self) -> str:
        return f"S{len(self.sources) + 1}"
