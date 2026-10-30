"""initial schema: research_jobs, chunks (pgvector), timeline_events

Revision ID: 0001
Revises:
Create Date: 2026-10-25
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from research_agent.db.models import Embedding

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "research_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("objective", sa.Text()),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("model", sa.String(64)),
        sa.Column("plan", sa.JSON()),
        sa.Column("report", sa.Text()),
        sa.Column("evaluation", sa.JSON()),
        sa.Column("result", sa.JSON()),
        sa.Column("error", sa.Text()),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("auto_approve", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("client_ip", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_research_jobs_status", "research_jobs", ["status"])
    op.create_index("ix_research_jobs_created_at", "research_jobs", ["created_at"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("research_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.String(64), nullable=False),
        sa.Column("source_id", sa.String(16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Embedding(), nullable=False),
    )
    op.create_index("ix_chunks_job_id", "chunks", ["job_id"])
    op.create_index("ix_chunks_job_chunk", "chunks", ["job_id", "chunk_id"], unique=True)

    op.create_table(
        "timeline_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("research_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("node", sa.String(64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("llm_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_timeline_events_job_id", "timeline_events", ["job_id"])


def downgrade() -> None:
    op.drop_table("timeline_events")
    op.drop_table("chunks")
    op.drop_table("research_jobs")
