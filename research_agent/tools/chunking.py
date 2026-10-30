"""Section-aware chunking.

The cleaned Markdown is split at headings first, then each section's paragraphs are
packed into chunks of about `max_tokens` with `overlap` tokens carried over from the
previous chunk in the same section. Chunks never straddle two sections.
"""

from __future__ import annotations

import re
from functools import lru_cache

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


@lru_cache
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline without a cached encoding: fall back to words
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    return len(enc.encode(text)) if enc else int(len(text.split()) * 1.3)


def _tail(text: str, n_tokens: int) -> str:
    enc = _encoder()
    if enc:
        return enc.decode(enc.encode(text)[-n_tokens:])
    words = text.split()
    return " ".join(words[-int(n_tokens / 1.3) :])


def _split_long(paragraph: str, max_tokens: int) -> list[str]:
    """Breaks one oversized paragraph at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts, cur = [], ""
    for s in sentences:
        if cur and count_tokens(cur + " " + s) > max_tokens:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Returns [(heading, body)], with "" as the heading of any leading text."""
    sections, last_heading, last_end = [], "", 0
    for m in _HEADING.finditer(markdown):
        body = markdown[last_end : m.start()].strip()
        if body:
            sections.append((last_heading, body))
        last_heading, last_end = m.group(2).strip(), m.end()
    tail = markdown[last_end:].strip()
    if tail:
        sections.append((last_heading, tail))
    return sections


def chunk_markdown(markdown: str, max_tokens: int = 450, overlap: int = 50) -> list[tuple[str, str]]:
    """Returns [(section_heading, chunk_text)]."""
    chunks: list[tuple[str, str]] = []
    for heading, body in split_sections(markdown):
        paragraphs = []
        for p in re.split(r"\n\s*\n", body):
            p = p.strip()
            if not p:
                continue
            paragraphs.extend(_split_long(p, max_tokens) if count_tokens(p) > max_tokens else [p])
        cur = ""
        for p in paragraphs:
            if cur and count_tokens(cur + "\n\n" + p) > max_tokens:
                chunks.append((heading, cur))
                cur = _tail(cur, overlap) + "\n\n" + p if overlap else p
            else:
                cur = f"{cur}\n\n{p}" if cur else p
        if cur:
            chunks.append((heading, cur))
    # Drop fragments too small to carry a claim (e.g. a lone caption).
    return [(h, t) for h, t in chunks if count_tokens(t) >= 25]
