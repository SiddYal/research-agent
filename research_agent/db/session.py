from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from research_agent.config import get_settings

ROOT = Path(__file__).resolve().parents[2]


def ensure_sqlite_dir(url: str) -> None:
    """SQLite creates the file but not its folder (data/ is absent in a fresh clone)."""
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        ensure_sqlite_dir(url)
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

        return engine
    # A PgBouncer pooler (Neon's "-pooler" host) runs in transaction mode, where
    # server-side prepared statements can land on another backend.
    connect_args = {"prepare_threshold": None} if "-pooler." in url else {}
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, connect_args=connect_args)


@lru_cache
def get_engine() -> Engine:
    return _make_engine(get_settings().database_url)


@lru_cache
def _sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(url: str | None = None) -> None:
    """Bring the schema up to date with Alembic (the single source of truth)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url or get_settings().migration_url)
    cfg.attributes["configure_logger"] = False  # keep the app's logging config
    command.upgrade(cfg, "head")
