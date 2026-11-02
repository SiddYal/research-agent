from functools import lru_cache
from typing import Literal

from pydantic import field_validator
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

    # Follow-up questions on a finished report (each researches a smaller slice)
    followup_max_subquestions: int = 3
    followup_max_sources: int = 8

    # Storage. On Neon, DATABASE_URL is the pooled endpoint for app queries and
    # DATABASE_URL_UNPOOLED the direct one for migrations and the checkpointer.
    database_url: str = "sqlite:///./data/research.db"
    database_url_unpooled: str | None = None
    checkpoint_url: str | None = None  # defaults to a sibling of database_url
    reports_dir: str = "./data/reports"

    # Object storage for report artifacts (any S3-compatible service; `neon deploy`
    # writes the AWS_* variables). Off unless REPORTS_BUCKET is set.
    reports_bucket: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_endpoint_url_s3: str | None = None
    aws_region: str | None = None

    # Guardrails for the public API
    rate_limit: str = "5/minute"
    daily_cost_cap_usd: float = 2.00
    max_job_cost_usd: float = 0.50
    worker_threads: int = 2

    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("database_url", "database_url_unpooled", "checkpoint_url", mode="after")
    @classmethod
    def _psycopg_driver(cls, url: str | None) -> str | None:
        """Hosted Postgres (Neon, Render) hands out postgres:// URLs; SQLAlchemy needs the driver named."""
        if not url:
            return None
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url.removeprefix(prefix)
        return url

    @field_validator("reports_bucket", mode="after")
    @classmethod
    def _blank_is_none(cls, v: str | None) -> str | None:
        return v or None

    @property
    def migration_url(self) -> str:
        """Direct connection for schema changes (a pooler can't hold session state)."""
        return self.database_url_unpooled or self.database_url

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
