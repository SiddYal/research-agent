import os
import tempfile
from pathlib import Path

# Must run before research_agent is imported: point the app at a throwaway DB
# (CI overrides DATABASE_URL with a Postgres service to test the pgvector path).
_tmp = Path(tempfile.mkdtemp(prefix="research-agent-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp.as_posix()}/test.db")
os.environ["REPORTS_DIR"] = str(_tmp / "reports")
# Never reach real infrastructure from .env (e.g. Neon's direct URL or bucket);
# only what's set explicitly in the environment, as for DATABASE_URL.
os.environ.setdefault("DATABASE_URL_UNPOOLED", "")
os.environ.setdefault("REPORTS_BUCKET", "")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ["LLM_PROVIDER"] = "openai"

import pytest  # noqa: E402

from research_agent.config import Settings, get_settings  # noqa: E402
from research_agent.db import init_db  # noqa: E402
from research_agent.db.vectors import MemoryVectorStore  # noqa: E402
from research_agent.llm import HashingEmbedder  # noqa: E402
from research_agent.nodes import Deps  # noqa: E402
from research_agent.state import ResearchState  # noqa: E402
from tests.fakes import FakeFetcher, FakeLLM, FakeSearch  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _db():
    init_db()


@pytest.fixture
def settings() -> Settings:
    return get_settings().model_copy(update={"max_verification_rounds": 1, "llm_sanitizer": True})


@pytest.fixture
def deps(settings) -> Deps:
    return Deps(
        llm=FakeLLM(),
        embedder=HashingEmbedder(256),
        store=MemoryVectorStore(),
        search=FakeSearch(),
        fetcher=FakeFetcher(),
        settings=settings,
    )


@pytest.fixture
def state() -> ResearchState:
    return ResearchState(job_id="job-test", question="What are the key facts about the Eiffel Tower?")
