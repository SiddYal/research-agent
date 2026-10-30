"""Search (collect candidate sources) and Ingest (fetch, clean, sanitize, chunk, embed)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from research_agent.logging import get_logger
from research_agent.nodes.base import Deps, traced
from research_agent.state import Chunk, ResearchState, Source
from research_agent.tools.chunking import chunk_markdown
from research_agent.tools.quality import source_quality
from research_agent.tools.sanitize import sanitize_text
from research_agent.tools.search import SKIP_DOMAINS, domain_of, normalize_url

log = get_logger(__name__)


def plausible_date(published: str | None) -> str | None:
    """Drops extractor artefacts such as '2000-01-01' or dates in the future."""
    from datetime import date

    if not published or not published[:4].isdigit():
        return None
    year = int(published[:4])
    return published if 2005 <= year <= date.today().year else None


def collect_sources(
    deps: Deps,
    queries: list[tuple[str, str]],  # (subquestion_id, query)
    existing: list[Source],
    limit: int,
    per_query: int,
    errors: list[str],
) -> list[Source]:
    """Runs the queries and returns new, de-duplicated sources, spread across subquestions."""
    seen = {normalize_url(s.url) for s in existing}
    buckets: dict[str, list[Source]] = {}
    for sq_id, query in queries:
        try:
            results = deps.search.search(query, per_query)
        except Exception as e:
            errors.append(f"search failed for {query!r}: {e!r}"[:300])
            log.warning("search_failed", query=query, error=repr(e))
            continue
        for r in results:
            key, domain = normalize_url(r.url), domain_of(r.url)
            if key in seen or domain in SKIP_DOMAINS or not r.url.startswith("http"):
                continue
            seen.add(key)
            buckets.setdefault(sq_id, []).append(
                Source(id="", url=r.url, title=r.title, snippet=r.snippet, query=query,
                       subquestion_id=sq_id, domain=domain, quality=source_quality(domain))
            )
    # Round-robin across subquestions so one broad query can't crowd out the rest.
    picked: list[Source] = []
    while len(picked) < limit and any(buckets.values()):
        for sq_id in list(buckets):
            if buckets[sq_id] and len(picked) < limit:
                picked.append(buckets[sq_id].pop(0))
    for i, s in enumerate(picked, len(existing) + 1):
        s.id = f"S{i}"
    return picked


def ingest_sources(deps: Deps, job_id: str, sources: list[Source]) -> tuple[list[Source], list[Chunk]]:
    """Fetches, cleans, sanitizes, chunks and embeds the given sources."""
    s = deps.settings
    with ThreadPoolExecutor(max_workers=8) as pool:
        pages = list(pool.map(lambda src: deps.fetcher.fetch(src.url), sources))

    sanitizer_llm = deps.llm if s.llm_sanitizer else None
    out_sources, chunks = [], []
    for src, page in zip(sources, pages):
        src = src.model_copy()
        if page:
            src.fetched = True
            # The page's own title beats the search engine's, which can be a mash-up.
            src.title = (page.title or src.title or "")[:150]
            src.published = plausible_date(page.published)
            raw = page.text
        else:
            raw = src.snippet  # fall back to the search snippet
        text, flags = sanitize_text(raw, sanitizer_llm)
        src.injection_flags = flags
        if flags:  # a page that tries to steer the agent is a weaker witness, too
            src.quality = round(src.quality * 0.5, 3)
            log.warning("injection_flagged", source=src.id, url=src.url, count=len(flags))
        pieces = chunk_markdown(text, s.chunk_tokens, s.chunk_overlap_tokens) if page else [("", text)]
        for n, (section, chunk_text) in enumerate(pieces):
            if chunk_text.strip() and chunk_text.strip() != "[removed: suspected prompt-injection text]":
                chunks.append(Chunk(id=f"{src.id}-c{n}", source_id=src.id, text=chunk_text,
                                    section=section, position=n))
        out_sources.append(src)

    if chunks:
        titles = {src.id: src.title for src in out_sources}
        vectors = deps.embedder.embed(
            [f"{titles[c.source_id]}\n{c.section}\n{c.text}" for c in chunks]
        )
        deps.store.add(job_id, [(c.id, c.source_id, c.text, v) for c, v in zip(chunks, vectors)])
    return out_sources, chunks


@traced("search")
def search_node(state: ResearchState, deps: Deps) -> dict:
    s = deps.settings
    queries = [(sq.id, q) for sq in state.plan.subquestions for q in sq.search_queries]
    errors = list(state.errors)
    sources = collect_sources(deps, queries, [], s.max_sources, s.results_per_query, errors)
    if not sources:
        raise RuntimeError("web search returned no usable sources")
    return {"sources": sources, "errors": errors}


@traced("ingest")
def ingest_node(state: ResearchState, deps: Deps) -> dict:
    sources, chunks = ingest_sources(deps, state.job_id, state.sources)
    if not chunks:
        raise RuntimeError("no text could be extracted from any source")
    log.info("ingested", fetched=sum(s.fetched for s in sources), sources=len(sources), chunks=len(chunks))
    return {"sources": sources, "chunks": chunks}
