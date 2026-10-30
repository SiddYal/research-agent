"""Page fetching and HTML cleaning.

trafilatura strips navigation, ads, cookie banners and other boilerplate and keeps
the main article text as Markdown, so headings survive for section-aware chunking.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import trafilatura

from research_agent.logging import get_logger
from research_agent.resilience import with_retry

log = get_logger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0; +https://github.com/)",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
    "Accept-Language": "en-US,en;q=0.8",
}


@dataclass
class FetchedPage:
    url: str
    text: str
    title: str | None = None
    published: str | None = None


class PageFetcher:
    def __init__(self, timeout_s: float = 15.0, max_chars: int = 40_000):
        self.timeout_s = timeout_s
        self.max_chars = max_chars

    @with_retry(max_attempts=2, max_wait_s=5)
    def _get(self, url: str) -> httpx.Response:
        resp = httpx.get(url, headers=HEADERS, timeout=self.timeout_s, follow_redirects=True)
        resp.raise_for_status()
        return resp

    def fetch(self, url: str) -> FetchedPage | None:
        """Returns None when the page can't be fetched or has no extractable text."""
        try:
            resp = self._get(url)
        except Exception as e:
            log.info("fetch_failed", url=url, error=repr(e)[:200])
            return None
        if "html" not in resp.headers.get("content-type", "html"):
            log.info("fetch_skipped_non_html", url=url, content_type=resp.headers.get("content-type"))
            return None
        return clean_html(resp.text, url, self.max_chars)


def clean_html(html: str, url: str | None = None, max_chars: int = 40_000) -> FetchedPage | None:
    text = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_tables=True,
        include_comments=False,
        include_links=False,
        favor_precision=True,
    )
    if not text or len(text) < 200:
        return None
    meta = trafilatura.extract_metadata(html)
    return FetchedPage(
        url=url or "",
        text=text[:max_chars],
        title=getattr(meta, "title", None) if meta else None,
        published=getattr(meta, "date", None) if meta else None,
    )
