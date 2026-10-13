"""add the missing HNSW index on claims.embedding

Revision ID: b7d2f4a8c1e9
Revises: a4c7e1f9b2d6
Create Date: 2026-09-25
"""

from alembic import op

revision = "b7d2f4a8c1e9"
down_revision = "a4c7e1f9b2d6"
branch_labels = None
depends_on = None

# ARCHITECTURE.md has always claimed an HNSW index on claims.embedding, but no
# migration ever created one — so the claim-graph LATERAL KNN fallback
# (api/v1/graph.py) and the verdict neighbour search (scripts/run_verdicts.py)
# ran as sequential scans. pgvector >= 0.5 is required for HNSW.
_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_claims_embedding_hnsw "
    "ON claims USING hnsw (embedding vector_cosine_ops)"
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(_INDEX)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_claims_embedding_hnsw")
