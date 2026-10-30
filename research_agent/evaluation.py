"""Automatic report evaluation: citation coverage, source diversity, hallucination
risk (LLM-as-judge against the verified claims), latency and cost."""

from __future__ import annotations

import re

from pydantic import BaseModel

from research_agent.llm import LLM
from research_agent.logging import get_logger
from research_agent.state import Evaluation, ResearchState

log = get_logger(__name__)

_CITE_GROUP = re.compile(r"\[((?:S\d+\s*,?\s*)+)\]")
_APPENDIX = re.compile(r"^##\s+(Contested and unverified claims|References)\b", re.M)


def cited_source_ids(text: str) -> list[str]:
    ids: list[str] = []
    for group in _CITE_GROUP.findall(text):
        for sid in re.findall(r"S\d+", group):
            if sid not in ids:
                ids.append(sid)
    return ids


def report_body(report: str) -> str:
    m = _APPENDIX.search(report)
    return report[: m.start()] if m else report


def statements(body: str) -> list[str]:
    """Sentences that could carry a factual statement (headings and fragments dropped)."""
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^([-*+]|\d+\.)\s+", "", line)
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", line):
            if len(sent.split()) >= 6:
                out.append(sent.strip())
    return out


class _Judged(BaseModel):
    index: int
    supported: bool


class _Judgement(BaseModel):
    statements: list[_Judged]


JUDGE_SYSTEM = (
    "You audit a research report for hallucinations. Given numbered report statements and the "
    "list of verified claims the writer was allowed to use, mark each statement supported=true if "
    "every factual assertion in it is backed by the claims (paraphrase and synthesis are fine), or "
    "if it makes no factual assertion (e.g. a transition or a statement about the evidence). Mark "
    "supported=false if it asserts any fact, figure or name not found in the claims. Judge every index."
)


def evaluate(state: ResearchState, llm: LLM | None) -> Evaluation:
    report = state.report or ""
    body = report_body(report)
    sents = statements(body)
    cited = [s for s in sents if _CITE_GROUP.search(s)]
    ids = cited_source_ids(body)
    known = {s.id: s for s in state.sources}
    valid = [i for i in ids if i in known]
    domains = {known[i].domain for i in valid}

    unsupported: list[str] = []
    verified = [c for c in state.verified_claims if c.status == "verified"]
    if llm is not None and sents and verified:
        claims = "\n".join(f"- {c.text}" for c in verified)
        numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
        try:
            out = llm.structured(
                JUDGE_SYSTEM, f"Verified claims:\n{claims}\n\nReport statements:\n{numbered}",
                _Judgement, max_tokens=2500,
            )
            unsupported = [sents[j.index] for j in out.statements if not j.supported and 0 <= j.index < len(sents)]
        except Exception as e:
            log.warning("hallucination_judge_failed", error=repr(e))

    return finalize_totals(Evaluation(
        citation_coverage=round(len(cited) / len(sents), 3) if sents else 0.0,
        cited_sources=len(valid),
        unique_domains=len(domains),
        source_diversity=round(len(domains) / len(valid), 3) if valid else 0.0,
        invalid_citations=[i for i in ids if i not in known],
        hallucination_risk=round(len(unsupported) / len(sents), 3) if sents else 0.0,
        unsupported_statements=unsupported,
        claims_total=len(state.verified_claims),
        claims_verified=len(verified),
        claims_uncertain=sum(c.status == "uncertain" for c in state.verified_claims),
        claims_contradicted=sum(c.status == "contradicted" for c in state.verified_claims),
        injection_flags=sum(len(s.injection_flags) for s in state.sources),
    ), state)


def finalize_totals(evaluation: Evaluation, state: ResearchState) -> Evaluation:
    """Latency, cost and token totals from the timeline as it stands.

    Latency is processing time (sum of node durations), so time spent waiting for a
    human to approve the plan isn't counted against the agent.
    """
    timeline = state.timeline
    latency = sum(e.duration_ms for e in timeline if e.node != "human_approval") / 1000
    return evaluation.model_copy(update={
        "latency_s": round(latency, 2),
        "cost_usd": round(state.cost_usd, 5),
        "tokens": sum(e.tokens for e in timeline),
        "llm_calls": sum(e.llm_calls for e in timeline),
    })
