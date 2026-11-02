from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from research_agent.logging import get_logger
from research_agent.nodes.base import Deps, traced
from research_agent.state import ResearchState, Source, VerifiedClaim

log = get_logger(__name__)

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


# ------------------------------------------------------------ follow-up edits

class _Edit(BaseModel):
    action: Literal["append", "insert", "replace"]
    section: str = Field(description="append/replace: the exact heading of the section to change. "
                                     "insert: the exact heading of the section the new one goes after")
    heading: str = Field(default="", description="insert only: the new section's heading, without '##'")
    content: str = Field(description="append: only the new paragraph(s). insert: the new section's text. "
                                     "replace: the section's full rewritten text. Never the heading line")


class _Revision(BaseModel):
    edits: list[_Edit]
    summary: str = Field(description="One or two sentences for the reader: what changed and where")


REVISE_SYSTEM = """You are a research analyst updating an existing report so it also answers a reader's
follow-up question, using a set of NEW verified claims. Edit the report; don't rewrite it.
Rules:
- Every NEW claim that helps answer the follow-up should end up in the report. Change only the
  sections that need it, with these edits:
  "append" adds new paragraph(s) to the end of an existing section: the usual way to extend one.
  "insert" adds a new "##" section after the one you name, when the follow-up needs its own section.
  "replace" rewrites a section in full; use it only when a new claim corrects something the section
  says. The rewrite must keep every other fact and citation the section had.
- Every new factual sentence must come from the NEW claims and end with their source ids, e.g.
  "The tower gets about 7 million visitors a year [S16]." Never invent ids or add outside facts.
- If a new claim contradicts something the report says, correct that sentence and say so plainly.
- Update the executive summary only if the follow-up changes its main points.
- Use the exact headings shown. Don't touch the title, and don't write a references list.
Today's date is {today}."""


def split_sections(body: str) -> tuple[str, list[tuple[str, str]]]:
    """'# Title' preamble, then (heading, content) for each '## ' section, in order."""
    preamble, sections = [], []
    for line in body.strip().splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
        else:
            preamble.append(line)
    return "\n".join(preamble).strip(), [(h, "\n".join(c).strip()) for h, c in sections]


def join_sections(preamble: str, sections: list[tuple[str, str]]) -> str:
    return "\n\n".join([preamble] + [f"## {h}\n\n{c}" for h, c in sections]).strip()


def _norm(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", heading.lower().lstrip("#")).strip()


def apply_edits(body: str, edits: list[_Edit]) -> tuple[str, list[str]]:
    """Applies section edits; untouched sections stay byte-for-byte. Returns (body, changed headings)."""
    preamble, sections = split_sections(body)
    changed: list[str] = []
    index = lambda h: next((i for i, (s, _) in enumerate(sections) if _norm(s) == _norm(h)), None)  # noqa: E731
    for e in edits:
        content = e.content.strip()
        if not content:
            continue
        i = index(e.section)
        if e.action in ("replace", "append") and i is not None:
            heading, existing = sections[i]
            if content == existing or e.action == "append" and content in existing:
                continue
            sections[i] = (heading, content if e.action == "replace" else f"{existing}\n\n{content}".strip())
            changed.append(heading)
            continue
        heading = (e.heading or e.section).strip().lstrip("#").strip()
        if not heading or index(heading) is not None and e.action == "insert":
            heading = f"{heading or 'Follow-up'} (follow-up)"
        # New sections go after the named one, or before the conclusion if it isn't found.
        if i is None:
            conclusion = index("Conclusion")
            i = (conclusion - 1) if conclusion is not None else len(sections) - 1
        sections.insert(i + 1, (heading, content))
        changed.append(heading)
    return join_sections(preamble, sections), changed


def dropped_citations(before: str, after: str) -> dict[str, list[str]]:
    """Source ids a section cited before an edit but not after: a sign the edit lost content."""
    old = dict(split_sections(before)[1])
    new = {_norm(h): c for h, c in split_sections(after)[1]}
    lost = {}
    for heading, content in old.items():
        gone = [s for s in _cited_ids(content) if s not in _cited_ids(new.get(_norm(heading), ""))]
        if gone:
            lost[heading] = gone
    return lost


def _is_replace_of(edit: _Edit, sections: dict[str, list[str]]) -> bool:
    return edit.action == "replace" and _norm(edit.section) in {_norm(h) for h in sections}


def revise_report(state: ResearchState, deps: Deps) -> dict:
    """Edits the existing report with a follow-up's new verified claims."""
    from research_agent.evaluation import report_body

    fu = state.pending_followup()
    new_ids = set(fu.subquestion_ids)
    new = [c for c in state.verified_claims if c.subquestion_id in new_ids and c.status == "verified"]
    body = report_body(state.report).strip()
    changed: list[str] = []
    if not new:
        revised = body
        summary = ("No new claims could be verified against independent sources, so the report's text "
                   "wasn't changed. Any unverified findings are listed under contested claims.")
    else:
        preamble, sections = split_sections(body)
        shown = "\n\n".join(f"## {h}\n{c}" for h, c in sections)
        user = (f"Original research question: {state.question}\nFollow-up question: {fu.question}\n\n"
                f"Current report (title: {preamble.lstrip('# ').strip()}):\n\n{shown}\n\n"
                f"NEW verified claims:\n{format_claims(new)}")
        system = REVISE_SYSTEM.format(today=date.today().isoformat())
        out = deps.llm.structured(system, user, _Revision, max_tokens=4000)
        if not apply_edits(body, out.edits)[1]:
            log.warning("revision_no_change", new_claims=len(new))
            out = deps.llm.structured(
                system, f"{user}\n\nYour edits didn't change the report. Work the NEW claims into it: "
                "\"append\" them to the most relevant section, or \"insert\" a new one.", _Revision,
                max_tokens=4000)
        edits = out.edits
        lost = dropped_citations(body, apply_edits(body, edits)[0])
        if lost:
            # A rewritten section lost some of what it said. Redo just those sections, naming what
            # went missing; the rest of the edit (e.g. a new section) stands.
            log.warning("revision_dropped_citations", sections=lost)
            problems = "; ".join(f'"{h}" no longer cites {", ".join(ids)}' for h, ids in lost.items())
            kept = [e.heading or e.section for e in edits if not _is_replace_of(e, lost)]
            retry = deps.llm.structured(
                system, f"{user}\n\nYour edit to these sections removed existing content: {problems}. "
                "Return a \"replace\" edit for each of them only: the section's existing content, every fact "
                "and citation, plus whatever new material belongs in that section. "
                + (f"Your other edits ({'; '.join(kept)}) are kept as they are, so don't repeat their content."
                   if kept else ""), _Revision, max_tokens=4000)
            redone = {_norm(e.section): e for e in retry.edits if e.action == "replace"}
            edits = [redone.get(_norm(e.section), e) if _is_replace_of(e, lost) else e for e in edits]
            still = dropped_citations(body, apply_edits(body, edits)[0])
            if still:  # keep those sections as they were rather than lose content
                log.warning("revision_section_kept", sections=still)
                edits = [e for e in edits if not _is_replace_of(e, still)]
        revised, changed = apply_edits(body, edits)
        if not changed:
            summary = "The new findings didn't change the report's text."
        elif edits is out.edits:
            summary = out.summary.strip()
        else:  # some of the edit was redone or left out, so the model's own summary may not fit
            summary = f"Updated: {', '.join(dict.fromkeys(changed))}."

    report = _with_appendix(state, revised)  # contested claims and references now include the new ones
    versions = state.report_versions + [state.report] if report != state.report else state.report_versions
    followups = list(state.followups)
    followups[-1] = fu.model_copy(update={
        "status": "completed", "summary": summary, "changed_sections": list(dict.fromkeys(changed)),
        "version": len(versions) + 1 if report != state.report else None})
    return {"report": report, "report_versions": versions, "followups": followups, "active": []}


def _with_appendix(state: ResearchState, body: str) -> str:
    return f"{body.strip()}\n\n{appendix(state, body)}\n"


@traced("writer")
def writer_node(state: ResearchState, deps: Deps) -> dict:
    if state.pending_followup():
        return revise_report(state, deps)
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
