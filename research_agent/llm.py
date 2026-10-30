"""Provider-neutral LLM access with cost metering.

Nodes only see the `LLM` protocol: `structured()` for schema-validated output,
`text()` for prose and `embed()` for vectors. Every call records its token usage
into the active `UsageMeter`, which is how per-node cost lands in the timeline and
how the per-job budget is enforced.
"""

from __future__ import annotations

import contextvars
import hashlib
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel

from research_agent.config import Settings, get_settings
from research_agent.logging import get_logger
from research_agent.resilience import with_retry

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (input, output). Unknown models are metered at zero with a warning.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class UsageMeter:
    budget_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    parents: tuple["UsageMeter", ...] = ()
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        price = PRICES.get(model)
        if price is None:
            log.warning("unknown_model_price", model=model)
            price = (0.0, 0.0)
        cost = (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
        chain = (self, *self.parents)
        for m in chain:
            with m._lock:
                m.input_tokens += input_tokens
                m.output_tokens += output_tokens
                m.cost_usd += cost
                m.calls += 1
        for m in chain:
            if m.budget_usd is not None and m.cost_usd > m.budget_usd:
                raise BudgetExceeded(f"job cost ${m.cost_usd:.4f} exceeded budget ${m.budget_usd:.2f}")


_meter: contextvars.ContextVar[UsageMeter | None] = contextvars.ContextVar("usage_meter", default=None)


class metered:
    """Context manager that routes usage into a fresh meter (and to any enclosing ones)."""

    def __init__(self, budget_usd: float | None = None, base_cost_usd: float = 0.0):
        self.meter = UsageMeter(budget_usd=budget_usd, cost_usd=base_cost_usd)

    def __enter__(self) -> UsageMeter:
        outer = _meter.get()
        if outer is not None:
            self.meter.parents = (outer, *outer.parents)
        self._token = _meter.set(self.meter)
        return self.meter

    def __exit__(self, *exc) -> None:
        _meter.reset(self._token)


def _record(model: str, input_tokens: int, output_tokens: int) -> None:
    meter = _meter.get()
    if meter is not None:
        meter.record(model, input_tokens, output_tokens)


class LLM(Protocol):
    model: str

    def structured(self, system: str, user: str, schema: type[T], max_tokens: int = 4000) -> T: ...

    def text(self, system: str, user: str, max_tokens: int = 4000) -> str: ...


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAILLM:
    def __init__(self, settings: Settings):
        import openai

        self.model = settings.openai_model
        self.client = openai.OpenAI(
            api_key=settings.openai_api_key, timeout=settings.llm_timeout_s, max_retries=0
        )
        self._retry = with_retry(settings.llm_max_attempts)

    def structured(self, system: str, user: str, schema: type[T], max_tokens: int = 4000) -> T:
        @self._retry
        def call():
            return self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format=schema,
                max_completion_tokens=max_tokens,
                temperature=0.2,
            )

        resp = call()
        _record(self.model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        msg = resp.choices[0].message
        if msg.parsed is None:
            raise ValueError(f"model returned no parsable output (refusal={msg.refusal!r})")
        return msg.parsed

    def text(self, system: str, user: str, max_tokens: int = 4000) -> str:
        @self._retry
        def call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=max_tokens,
                temperature=0.3,
            )

        resp = call()
        _record(self.model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content or ""


class AnthropicLLM:
    def __init__(self, settings: Settings):
        import anthropic

        self.model = settings.anthropic_model
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_s, max_retries=0
        )
        self._retry = with_retry(settings.llm_max_attempts)

    def _check(self, resp) -> None:
        if resp.stop_reason == "refusal":
            raise ValueError(f"model declined the request: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise ValueError("model output hit max_tokens before completing")

    def structured(self, system: str, user: str, schema: type[T], max_tokens: int = 4000) -> T:
        @self._retry
        def call():
            return self.client.messages.parse(
                model=self.model,
                max_tokens=max_tokens + 4000,  # headroom for adaptive thinking
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )

        resp = call()
        _record(self.model, resp.usage.input_tokens, resp.usage.output_tokens)
        self._check(resp)
        if resp.parsed_output is None:
            raise ValueError("model returned no parsable output")
        return resp.parsed_output

    def text(self, system: str, user: str, max_tokens: int = 4000) -> str:
        @self._retry
        def call():
            return self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens + 4000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )

        resp = call()
        _record(self.model, resp.usage.input_tokens, resp.usage.output_tokens)
        self._check(resp)
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAIEmbedder:
    def __init__(self, settings: Settings):
        import openai

        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self.client = openai.OpenAI(
            api_key=settings.openai_api_key, timeout=settings.llm_timeout_s, max_retries=0
        )
        self._retry = with_retry(settings.llm_max_attempts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):
            batch = [t[:8000] or " " for t in texts[i : i + 96]]
            resp = self._retry(self.client.embeddings.create)(
                model=self.model, input=batch, dimensions=self.dim
            )
            _record(self.model, resp.usage.prompt_tokens, 0)
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out


class HashingEmbedder:
    """Dependency-free fallback: signed feature hashing of word uni/bigrams.

    Much weaker than a learned embedding, but deterministic and free, which makes it
    the right choice for tests and for running without an OpenAI key.
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = re.findall(r"[a-z0-9]+", text.lower())
        for gram in words + [f"{a} {b}" for a, b in zip(words, words[1:])]:
            h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@dataclass(frozen=True)
class ModelOption:
    id: str  # "<provider>/<model>", what clients send and jobs store
    provider: str
    model: str
    label: str


# Models offered to users. A provider's models are listed only when its API key is set.
CATALOG = [
    ModelOption("openai/gpt-4.1-mini", "openai", "gpt-4.1-mini", "OpenAI · GPT-4.1 mini"),
    ModelOption("anthropic/claude-haiku-5-5", "anthropic", "claude-haiku-5-5", "Anthropic · Claude Haiku 5.5"),
]


def available_models(settings: Settings | None = None) -> list[ModelOption]:
    """Usable models, with the configured default first."""
    settings = settings or get_settings()
    keys = {"openai": settings.openai_api_key, "anthropic": settings.anthropic_api_key}
    options = [m for m in CATALOG if keys.get(m.provider)]
    try:
        default_id = f"{settings.provider}/{settings.model}"
    except RuntimeError:  # no key configured at all
        return options
    if not any(m.id == default_id for m in options):  # a model set via env but not in the catalog
        options.append(ModelOption(default_id, settings.provider, settings.model, f"{settings.provider} · {settings.model}"))
    return sorted(options, key=lambda m: m.id != default_id)


def make_llm(settings: Settings | None = None, option: ModelOption | None = None) -> LLM:
    settings = settings or get_settings()
    if option is not None:
        field = "openai_model" if option.provider == "openai" else "anthropic_model"
        settings = settings.model_copy(update={"llm_provider": option.provider, field: option.model})
    return OpenAILLM(settings) if settings.provider == "openai" else AnthropicLLM(settings)


def make_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.openai_api_key:
        return OpenAIEmbedder(settings)
    return HashingEmbedder(settings.embedding_dim)
