"""Source quality and freshness heuristics used by the verifier."""

from __future__ import annotations

import re
from datetime import date

HIGH = {
    "wikipedia.org", "arxiv.org", "nature.com", "science.org", "nih.gov", "who.int",
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com", "ft.com",
    "economist.com", "acm.org", "ieee.org", "springer.com", "sciencedirect.com",
    "github.com", "docs.python.org", "developer.mozilla.org", "openai.com", "anthropic.com",
    "aws.amazon.com", "cloud.google.com", "learn.microsoft.com", "kubernetes.io", "mckinsey.com",
    "gartner.com", "statista.com", "brookings.edu", "pewresearch.org", "britannica.com",
}
LOW = {"quora.com", "reddit.com", "medium.com", "pinterest.com", "answers.com", "fandom.com"}


def source_quality(domain: str) -> float:
    d = domain.lower()
    if d.endswith((".gov", ".edu", ".mil")) or ".gov." in d or ".ac." in d:
        return 0.9
    if any(d == h or d.endswith("." + h) for h in HIGH):
        return 0.85
    if any(d == l or d.endswith("." + l) for l in LOW):
        return 0.35
    return 0.55


def age_years(published: str | None, today: date | None = None) -> float | None:
    if not published:
        return None
    m = re.match(r"(\d{4})(?:-(\d{2}))?", published)
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2) or 6)
    today = today or date.today()
    return max(0.0, (today.year - year) + (today.month - month) / 12)
