"""Claim extraction, verification, and the follow-up retrieval loop.

The writer never sees raw retrieved text: it only receives claims that passed
verification here. Claims without enough support are routed to `followup_node`
for one more round of targeted search before being marked uncertain.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from research_agent.logging import get_logger
from research_agent.nodes.base import UNTRUSTED_NOTE, Deps, traced, wrap_source
from research_agent.nodes.retrieve import make_retriever
from research_agent.nodes.search import collect_sources, ingest_sources
from research_agent.state import Chunk, Claim, Evidence, ResearchState, Source, VerifiedClaim
from research_agent.tools.quality import age_years

log = get_logger(__name__)


# ---------------------------------------------------------------- extraction

class _ClaimOut(BaseModel):
    text: str = Field(description="One self-contained factual statement, understandable on its own")
    chunk_id: str = Field(description="id of the passage the claim comes from")
    entities: list[str]
    numbers: list[str]
    dates: list[str]
    confidence: float = Field(description="0-1: how clearly the passage states this")


class _ClaimsOut(BaseModel):
    claims: list[_ClaimOut]


EXTRACT_SYSTEM = f"""You extract atomic factual claims from source passages to answer a research subquestion.
Rules:
- Each claim is ONE checkable fact (not a paragraph), self-contained (resolve pronouns, name the subject),
  and stated only as strongly as the passage states it.
- Prefer specific claims: figures, dates, named entities, comparisons, findings.
- Only extract what the passage itself says; skip opinions, marketing language and speculation.
- Copy the chunk_id of the passage the claim came from exactly.
- Return at most {{max_claims}} claims, most useful first. Return none if nothing is relevant.
{UNTRUSTED_NOTE}"""


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_duplicate(text: str, seen: list[set[str]], threshold: float = 0.8) -> bool:
    """Near-duplicate check (Jaccard over word sets): overlapping subquestions often
    retrieve the same passage and would otherwise yield the same claim twice."""
    terms = _terms(text)
    return any(len(terms & s) / max(1, len(terms | s)) >= threshold for s in seen)


@traced("extract_claims")
def extract_node(state: ResearchState, deps: Deps) -> dict:
    chunk_map = state.chunk_map()
    claims: list[Claim] = list(state.extracted_claims)  # a follow-up adds to the earlier ones
    seen: list[set[str]] = [_terms(c.text) for c in claims]
    for sq in state.active_subquestions():
        chunks = [chunk_map[cid] for cid in state.retrieved.get(sq.id, []) if cid in chunk_map]
        if not chunks:
            continue
        passages = "\n\n".join(
            wrap_source(c.source_id, state.source(c.source_id).title, c.text, c.id) for c in chunks
        )
        out = deps.llm.structured(
            EXTRACT_SYSTEM.format(max_claims=deps.settings.max_claims_per_subquestion),
            f"Subquestion: {sq.question}\n\n{passages}",
            _ClaimsOut,
            max_tokens=2500,
        )
        for c in out.claims[: deps.settings.max_claims_per_subquestion]:
            chunk = chunk_map.get(c.chunk_id)
            if chunk is None:
                log.warning("claim_unknown_chunk", chunk_id=c.chunk_id)
                continue
            if is_duplicate(c.text, seen):
                continue
            seen.append(_terms(c.text))
            claims.append(Claim(
                id=f"C{len(claims) + 1}", text=c.text, source_id=chunk.source_id, chunk_id=chunk.id,
                subquestion_id=sq.id, entities=c.entities, numbers=c.numbers, dates=c.dates,
                confidence=max(0.0, min(1.0, c.confidence)),
            ))
    return {"extracted_claims": claims}


# -------------------------------------------------------------- verification

class _Assessment(BaseModel):
    chunk_id: str
    stance: Literal["supports", "contradicts", "irrelevant"]
    note: str = Field(description="Short reason, e.g. the conflicting figure")


class _ClaimVerdict(BaseModel):
    claim_id: str
    assessments: list[_Assessment]


class _Verdicts(BaseModel):
    verdicts: list[_ClaimVerdict]


VERIFY_SYSTEM = f"""You are a meticulous fact-checker. For each claim, judge every evidence passage listed under it:
- "supports": the passage states the same fact (allowing paraphrase and rounding),
- "contradicts": the passage states something incompatible (different figure, date, outcome, or a denial),
- "irrelevant": the passage doesn't address the claim.
Judge only from the passages, not from your own knowledge. Assess every passage of every claim.
{UNTRUSTED_NOTE}"""


def decide_status(claim: Claim, evidence: list[Evidence], sources: dict[str, Source],
                  stale_after_years: float = 3.0) -> VerifiedClaim:
    """Deterministic verdict from per-passage stances (kept out of the LLM so it's testable)."""
    origin_ok = not any(
        e.chunk_id == claim.chunk_id and e.stance != "supports" for e in evidence
    )
    support = {e.source_id for e in evidence if e.stance == "supports"}
    if origin_ok:
        support.add(claim.source_id)
    contra = {e.source_id for e in evidence if e.stance == "contradicts"} - support
    weight = lambda ids: sum(sources[s].quality if s in sources else 0.5 for s in ids)  # noqa: E731
    support_w, contra_w = weight(support), weight(contra)
    quality = max((sources[s].quality for s in support if s in sources), default=0.0)

    notes = []
    if not origin_ok:
        notes.append("the cited passage does not clearly state this claim")
    if contra and contra_w >= support_w:
        status = "contradicted"
        notes.append(f"contradicted by {', '.join(sorted(contra))}")
    elif contra and len(support) >= 2 and support_w >= 2 * contra_w:
        status = "verified"  # a clear, quality-weighted majority of independent sources
        notes.append(f"disputed by {', '.join(sorted(contra))}")
    elif contra:
        status = "uncertain"
        notes.append(f"contested: supported by {len(support)}, contradicted by {len(contra)} source(s)")
    elif len(support) >= 2:
        status = "verified"
    elif len(support) == 1 and quality >= 0.8:
        status = "verified"
        notes.append("single authoritative source")
    else:
        status = "uncertain"
        notes.append("not independently corroborated" if support else "no supporting evidence")

    confidence = 0.4 * claim.confidence + 0.15 * min(len(support), 3) + 0.2 * quality - 0.3 * len(contra)
    origin = sources.get(claim.source_id)
    freshness = origin.published if origin else None
    age = age_years(freshness)
    if age is not None and age > stale_after_years and (claim.numbers or claim.dates):
        notes.append(f"source is from {freshness[:4]}; figures may be out of date")
        confidence *= 0.85
    return VerifiedClaim(
        **claim.model_dump(), status=status, supporting_sources=sorted(support),
        contradicting_sources=sorted(contra), evidence=evidence, notes="; ".join(notes),
        freshness=freshness, final_confidence=round(max(0.0, min(1.0, confidence)), 3),
    )


def gather_evidence(claim: Claim, retriever, chunk_map: dict[str, Chunk], k: int = 5) -> list[Chunk]:
    """Origin passage plus the best other passages, at most two per source."""
    picked = [chunk_map[claim.chunk_id]] if claim.chunk_id in chunk_map else []
    per_source: dict[str, int] = {claim.source_id: 1}
    for c in retriever.search(claim.text, 15):
        if len(picked) >= k:
            break
        if c.id == claim.chunk_id or per_source.get(c.source_id, 0) >= 2:
            continue
        per_source[c.source_id] = per_source.get(c.source_id, 0) + 1
        picked.append(c)
    return picked


@traced("verify")
def verify_node(state: ResearchState, deps: Deps) -> dict:
    if state.verification_round == 0:
        # Claims verified in an earlier pass (before a reader's follow-up) keep their verdicts.
        done: dict[str, VerifiedClaim] = {c.id: c for c in state.verified_claims}
        todo: list[Claim] = [c for c in state.extracted_claims if c.id not in done]
    else:  # re-check only what follow-up retrieval was meant to resolve
        unresolved = {c.id for c in _unresolved(state)}
        done = {c.id: c for c in state.verified_claims if c.id not in unresolved}
        todo = [Claim(**c.model_dump(include=set(Claim.model_fields)))
                for c in state.verified_claims if c.id not in done]

    retriever = make_retriever(state, deps)
    chunk_map = state.chunk_map()
    sources = {s.id: s for s in state.sources}
    results = dict(done)

    previous = {c.id: c for c in state.verified_claims}
    if state.verification_round > 0:
        # Re-judge a claim only if follow-up retrieval actually changed its evidence;
        # asking again about the same passages would just re-roll the verdict.
        unchanged = []
        for c in todo:
            old = {e.chunk_id for e in previous[c.id].evidence} | {c.chunk_id}
            if {ch.id for ch in gather_evidence(c, retriever, chunk_map)} <= old:
                unchanged.append(c.id)
        for cid in unchanged:
            results[cid] = previous[cid]
        todo = [c for c in todo if c.id not in unchanged]

    for i in range(0, len(todo), 4):
        batch = todo[i : i + 4]
        evidence = {c.id: gather_evidence(c, retriever, chunk_map) for c in batch}
        blocks = []
        for c in batch:
            passages = "\n".join(
                wrap_source(ch.source_id, sources[ch.source_id].title, ch.text[:1500], ch.id)
                for ch in evidence[c.id]
            )
            blocks.append(f'<claim id="{c.id}">{c.text}</claim>\nEvidence:\n{passages}')
        out = deps.llm.structured(VERIFY_SYSTEM, "\n\n---\n\n".join(blocks), _Verdicts, max_tokens=2500)
        verdicts = {v.claim_id: v for v in out.verdicts}
        for c in batch:
            ev_chunks = {ch.id: ch for ch in evidence[c.id]}
            assessed = []
            for a in verdicts[c.id].assessments if c.id in verdicts else []:
                stance = a.stance.strip().lower()
                if a.chunk_id in ev_chunks and stance in ("supports", "contradicts", "irrelevant"):
                    assessed.append(Evidence(source_id=ev_chunks[a.chunk_id].source_id,
                                             chunk_id=a.chunk_id, stance=stance, note=a.note))
            results[c.id] = decide_status(c, assessed, sources)

    ordered = [results[c.id] for c in state.extracted_claims if c.id in results]
    counts = {k: sum(c.status == k for c in ordered) for k in ("verified", "uncertain", "contradicted")}
    log.info("verification_done", round=state.verification_round, **counts)
    return {"verified_claims": ordered}


def _unresolved(state: ResearchState) -> list[VerifiedClaim]:
    """Uncorroborated claims from the current pass (earlier passes' verdicts are final)."""
    active = {sq.id for sq in state.active_subquestions()}
    return [c for c in state.verified_claims
            if c.status == "uncertain" and not c.contradicting_sources and c.subquestion_id in active]


def needs_followup(state: ResearchState, max_rounds: int) -> bool:
    return bool(_unresolved(state)) and state.verification_round < max_rounds


def keyword_query(claim: Claim, max_terms: int = 9) -> str:
    """A search-engine-friendly query for a claim: its entities, then content words.
    (Whole claim sentences mostly return nothing.)"""
    from research_agent.tools.retrieval import tokenize

    terms: list[str] = []
    for word in " ".join(claim.entities).split() + tokenize(claim.text):
        if word.lower() not in {t.lower() for t in terms}:
            terms.append(word)
    return " ".join(terms[:max_terms])


@traced("followup_search")
def followup_node(state: ResearchState, deps: Deps) -> dict:
    """Targeted search for the weakest uncertain claims, to find corroborating sources."""
    unresolved = sorted(_unresolved(state), key=lambda c: c.final_confidence)[:4]
    errors = list(state.errors)
    queries = [(c.subquestion_id, keyword_query(c)) for c in unresolved]
    new = collect_sources(deps, queries, state.sources, limit=6, per_query=3, errors=errors)
    sources, chunks = list(state.sources), list(state.chunks)
    if new:
        ingested, new_chunks = ingest_sources(deps, state.job_id, new)
        sources += ingested
        chunks += new_chunks
    return {"sources": sources, "chunks": chunks, "errors": errors,
            "verification_round": state.verification_round + 1}
