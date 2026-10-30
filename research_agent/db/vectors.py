"""Chunk embedding storage and similarity search, scoped to one research job.

On Postgres the nearest-neighbour search runs in the database via pgvector's
cosine-distance operator; on SQLite the job's vectors are loaded and scored with
numpy (a job holds a few hundred chunks, so this stays fast).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sqlalchemy import Float, delete, literal, select

from research_agent.db.models import ChunkRecord
from research_agent.db.session import get_engine, session_scope


class VectorStore(Protocol):
    def add(self, job_id: str, items: list[tuple[str, str, str, list[float]]]) -> None: ...

    def search(self, job_id: str, query: list[float], k: int) -> list[tuple[str, float]]: ...


def _cosine_top_k(ids: list[str], mat: np.ndarray, query: list[float], k: int) -> list[tuple[str, float]]:
    if not ids:
        return []
    q = np.asarray(query, dtype=np.float32)
    sims = mat @ q / (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-9)
    order = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in order]


class SqlVectorStore:
    """items: (chunk_id, source_id, text, embedding)"""

    def add(self, job_id: str, items: list[tuple[str, str, str, list[float]]]) -> None:
        if not items:
            return
        with session_scope() as s:
            s.execute(
                delete(ChunkRecord).where(
                    ChunkRecord.job_id == job_id, ChunkRecord.chunk_id.in_([i[0] for i in items])
                )
            )
            s.add_all(
                ChunkRecord(job_id=job_id, chunk_id=c, source_id=src, text=t, embedding=e)
                for c, src, t, e in items
            )

    def search(self, job_id: str, query: list[float], k: int) -> list[tuple[str, float]]:
        with session_scope() as s:
            if get_engine().dialect.name == "postgresql":
                from pgvector.sqlalchemy import Vector

                q = literal(query, type_=Vector(len(query)))
                dist = ChunkRecord.embedding.op("<=>", return_type=Float())(q)
                rows = s.execute(
                    select(ChunkRecord.chunk_id, dist.label("d"))
                    .where(ChunkRecord.job_id == job_id)
                    .order_by(dist)
                    .limit(k)
                ).all()
                return [(cid, 1.0 - float(d)) for cid, d in rows]
            rows = s.execute(
                select(ChunkRecord.chunk_id, ChunkRecord.embedding).where(ChunkRecord.job_id == job_id)
            ).all()
        ids = [r[0] for r in rows]
        mat = np.asarray([r[1] for r in rows], dtype=np.float32) if rows else np.zeros((0, 1))
        return _cosine_top_k(ids, mat, query, k)


class MemoryVectorStore:
    """In-process store for tests and the DB-less CLI path."""

    def __init__(self):
        self._data: dict[str, dict[str, list[float]]] = {}

    def add(self, job_id: str, items: list[tuple[str, str, str, list[float]]]) -> None:
        bucket = self._data.setdefault(job_id, {})
        for chunk_id, _src, _text, emb in items:
            bucket[chunk_id] = emb

    def search(self, job_id: str, query: list[float], k: int) -> list[tuple[str, float]]:
        bucket = self._data.get(job_id, {})
        ids = list(bucket)
        mat = np.asarray([bucket[i] for i in ids], dtype=np.float32) if ids else np.zeros((0, 1))
        return _cosine_top_k(ids, mat, query, k)
