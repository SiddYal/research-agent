from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider. Left unset, it is picked from whichever API key is present.
    llm_provider: Literal["openai", "anthropic"] | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    anthropic_model: str = "claude-haiku-5-5"
    llm_timeout_s: float = 120.0
    llm_max_attempts: int = 4

    # Embeddings: OpenAI when a key is present, otherwise a local hashing embedder.
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Search
    tavily_api_key: str | None = None
    results_per_query: int = 5
    max_sources: int = 15
    fetch_timeout_s: float = 15.0
    max_doc_chars: int = 40_000

    # Planning / retrieval / verification
    max_subquestions: int = 5
    chunk_tokens: int = 450
    chunk_overlap_tokens: int = 50
    retrieval_candidates: int = 20
    top_k_chunks: int = 6
    reranker: Literal["llm", "none"] = "llm"
    max_claims_per_subquestion: int = 6
    max_verification_rounds: int = 1
    llm_sanitizer: bool = True

    # Storage
    database_url: str = "sqlite:///./data/research.db"
    checkpoint_url: str | None = None  # defaults to a sibling of database_url
    reports_dir: str = "./data/reports"

    # Guardrails for the public API
    rate_limit: str = "5/minute"
    daily_cost_cap_usd: float = 2.00
    max_job_cost_usd: float = 0.50
    worker_threads: int = 2

    log_level: str = "INFO"
    log_json: bool = False

    @property
    def provider(self) -> Literal["openai", "anthropic"]:
        if self.llm_provider:
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        raise RuntimeError("No LLM configured: set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

    @property
    def model(self) -> str:
        return self.openai_model if self.provider == "openai" else self.anthropic_model

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")


@lru_cache
def get_settings() -> Settings:
    return Settings()
