"""Web search. DuckDuckGo (via `ddgs`) needs no key; Tavily is used when a key is set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from research_agent.config import Settings, get_settings
from research_agent.logging import get_logger
from research_agent.resilience import TransientError, with_retry

log = get_logger(__name__)

# Pages that rarely contain extractable article text.
SKIP_DOMAINS = {
    "youtube.com", "m.youtube.com", "facebook.com", "instagram.com", "tiktok.com",
    "x.com", "twitter.com", "pinterest.com", "linkedin.com",
}


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower().removeprefix("www."), path, parts.query, ""))


def domain_of(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


class DuckDuckGoSearch:
    @with_retry(max_attempts=3)
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        try:
            raw = DDGS(timeout=15).text(query, max_results=max_results) or []
        except (RatelimitException, TimeoutException) as e:
            raise TransientError(str(e)) from e
        except DDGSException as e:  # "No results found" and similar
            log.info("search_no_results", query=query, error=str(e))
            return []
        return [SearchResult(r.get("title", ""), r["href"], r.get("body", "")) for r in raw if r.get("href")]


class TavilySearch:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @with_retry(max_attempts=3)
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": max_results},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=20,
        )
        resp.raise_for_status()
        return [
            SearchResult(r.get("title", ""), r["url"], r.get("content", ""))
            for r in resp.json().get("results", [])
        ]


def make_search(settings: Settings | None = None) -> SearchProvider:
    settings = settings or get_settings()
    return TavilySearch(settings.tavily_api_key) if settings.tavily_api_key else DuckDuckGoSearch()
