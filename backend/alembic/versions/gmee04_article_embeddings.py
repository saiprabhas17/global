"""Make article vectors and hybrid evidence search available after migration.

Revision ID: gmee04
Revises: gmee03

The legacy backfill created this column/index outside Alembic. IF NOT EXISTS
preserves those existing vectors while also supporting a fresh installation.
"""

from alembic import op

revision = "gmee04"
down_revision = "gmee03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS embedding vector(768)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS articles_embedding_ix "
        "ON articles USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS claims_text_search_ix "
        "ON claims USING gin (to_tsvector('english', claim_text))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS claims_text_search_ix")
    op.execute("DROP INDEX IF EXISTS articles_embedding_ix")
    op.drop_column("articles", "embedding")
