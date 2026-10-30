from __future__ import annotations

from datetime import date

from research_agent.nodes.base import Deps, traced
from research_agent.state import ResearchState, Source, VerifiedClaim

WRITER_SYSTEM = """You are a research analyst writing a report in Markdown from a set of VERIFIED claims.
Rules:
- Use ONLY the claims provided. Do not add facts, figures or examples from your own knowledge.
  You may add connective reasoning and synthesis, but every factual sentence must come from the claims.
- Cite inline, sentence by sentence: end EVERY sentence that states a fact with the source ids of the
  claims used in THAT sentence, e.g. "Pinecone charges $0.33 per GB of storage [S5]." Never collect
  citations at the end of a paragraph, and don't list a source for a sentence it doesn't support.
  This applies to bullet points and the executive summary too. Use only source ids listed on the
  claims you used. Never invent ids.
- Structure: "# <title>", then "## Executive summary" (3-5 sentences), then one "##" section per
  outline item, then "## Conclusion" including the main limitations of the evidence.
- Where claims are thin for a section, say so plainly instead of padding.
- Do not write a references list; it is appended automatically.
Today's date is {today}."""


def format_claims(claims: list[VerifiedClaim]) -> str:
    return "\n".join(
        f"- ({c.subquestion_id}) {c.text} [sources: {', '.join(c.supporting_sources)}]"
        f" (confidence {c.final_confidence:.2f}{'; ' + c.notes if c.notes else ''})"
        for c in claims
    )


def appendix(state: ResearchState, report_body: str) -> str:
    """Deterministic sections: unverified claims and references (never LLM-written)."""
    flagged = [c for c in state.verified_claims if c.status != "verified"]
    parts = []
    if flagged:
        parts.append("## Contested and unverified claims\n")
        parts.append("These claims were found in sources but did not pass verification, "
                     "so the report above does not rely on them.\n")
        for c in flagged:
            src = ", ".join(c.supporting_sources) or c.source_id
            parts.append(f"- **{c.status.upper()}**: {c.text} [{src}] — {c.notes or 'insufficient evidence'}")
        parts.append("")
    cited = sorted(
        {s for s in _cited_ids(report_body)} | {s for c in flagged for s in c.supporting_sources + c.contradicting_sources},
        key=lambda s: int(s[1:]),
    )
    parts.append("## References\n")
    for sid in cited:
        src: Source | None = state.source(sid)
        if src:
            date_s = f" ({src.published[:10]})" if src.published else ""
            parts.append(f"- [{sid}] {src.title or src.domain}{date_s}. <{src.url}>")
    return "\n".join(parts)


def _cited_ids(text: str) -> list[str]:
    from research_agent.evaluation import cited_source_ids

    return cited_source_ids(text)


@traced("writer")
def writer_node(state: ResearchState, deps: Deps) -> dict:
    verified = [c for c in state.verified_claims if c.status == "verified"]
    plan = state.plan
    if not verified:
        body = (
            f"# {state.question}\n\n## Executive summary\n\nNo claims could be verified against "
            "independent sources, so no findings are reported. See the unverified claims below "
            "and consider rephrasing or narrowing the question."
        )
    else:
        user = (
            f"Research question: {state.question}\nObjective: {plan.objective}\nScope: {plan.scope}\n"
            f"Outline: {'; '.join(plan.outline)}\n"
            f"Subquestions: {'; '.join(f'{q.id}: {q.question}' for q in plan.subquestions)}\n\n"
            f"Verified claims:\n{format_claims(verified)}"
        )
        body = deps.llm.text(WRITER_SYSTEM.format(today=date.today().isoformat()), user, max_tokens=4000).strip()
    return {"report": f"{body}\n\n{appendix(state, body)}\n"}
