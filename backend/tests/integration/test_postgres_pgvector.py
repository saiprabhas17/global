"""Integration tests against real Postgres + pgvector.

The unit suite runs on SQLite with dialect shims, which means the production SQL
that matters most is never exercised: the `<=>` cosine operator, the HNSW index
on claims.embedding, and the unique edge constraint added by migration
a4c7e1f9b2d6. These tests cover exactly that gap.

They skip when no Postgres is reachable during optional local runs. CI sets
GMEE_REQUIRE_INTEGRATION=1 so a missing database is a failure. The backend CI
job provides a fresh pgvector service, migrates it, then runs:

    pytest tests/integration -m integration
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
async def _require_postgres() -> AsyncGenerator[None, None]:
    """Skip when no database is reachable — and keep the pool loop-safe.

    pytest-asyncio hands every test a fresh event loop, but the module-level
    engine's pooled connections still belong to the previous loop; without the
    dispose they raise and the guard would silently skip healthy tests.
    """
    from app.db.postgres import async_session_maker, engine

    await engine.dispose()
    try:
        async with async_session_maker() as db:
            await db.execute(text("SELECT 1"))
    except Exception as exc:
        if os.environ.get("GMEE_REQUIRE_INTEGRATION") == "1":
            pytest.fail(f"Required Postgres not reachable ({type(exc).__name__})")
        pytest.skip(f"Postgres not reachable ({type(exc).__name__})")
    yield
    await engine.dispose()


async def _scalar(sql: str) -> Any:
    from app.db.postgres import async_session_maker

    async with async_session_maker() as db:
        return (await db.execute(text(sql))).scalar()


@pytest.mark.asyncio
async def test_pgvector_cosine_operator() -> None:
    distance = await _scalar("SELECT ('[1,0,0]'::vector <=> '[0,1,0]'::vector)")
    assert float(cast(float, distance)) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_hnsw_index_exists_on_claim_embeddings() -> None:
    """Migration b7d2f4a8c1e9 — the index ARCHITECTURE.md always assumed."""
    from app.db.postgres import async_session_maker

    async with async_session_maker() as db:
        rows: Any = (
            await db.execute(
                text("SELECT indexdef FROM pg_indexes WHERE tablename = 'claims'")
            )
        ).scalars().all()
    definitions = [str(d).lower() for d in rows]
    assert any(
        "hnsw" in d and "embedding" in d for d in definitions
    ), f"no HNSW index on claims.embedding: {definitions}"


@pytest.mark.asyncio
async def test_unique_edge_constraint_exists() -> None:
    n = await _scalar(
        "SELECT count(*) FROM pg_constraint "
        "WHERE conname = 'uq_claim_relationships_edge'"
    )
    assert int(cast(int, n)) == 1


@pytest.mark.asyncio
async def test_database_is_migrated_to_the_script_head() -> None:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    head = ScriptDirectory.from_config(cfg).get_current_head()

    version = await _scalar("SELECT version_num FROM alembic_version")
    assert version == head, "run `alembic upgrade head` against this database"


@pytest.mark.asyncio
async def test_duplicate_mutation_edge_is_rejected() -> None:
    """Check uniqueness on a fresh database, rolling back every fixture row."""
    from app.db.postgres import async_session_maker
    from app.models.article import Article
    from app.models.claim import Claim
    from app.models.evolution import ClaimRelationship, RelationshipTypeEnum
    from app.models.source import Source, SourceTypeEnum

    async with async_session_maker() as db:
        fixture_id = uuid.uuid4()
        source = Source(
            name=f"Integration fixture {fixture_id}",
            type=SourceTypeEnum.rss,
            url_or_identifier=f"https://example.invalid/integration/{fixture_id}",
            is_active=False,
        )
        db.add(source)
        await db.flush()
        article = Article(
            source_id=source.id,
            title="Synthetic database constraint fixture",
            url=f"https://example.invalid/integration/{fixture_id}/article",
            content_hash=fixture_id.hex,
            raw_metadata={"origin": "integration_test", "scientific_evidence": False},
        )
        db.add(article)
        await db.flush()
        claims = [
            Claim(
                article_id=article.id,
                claim_text=f"Synthetic integration constraint fixture {index}.",
            )
            for index in range(2)
        ]
        db.add_all(claims)
        await db.flush()
        from_id, to_id = (claim.id for claim in claims)
        db.add(ClaimRelationship(
            from_claim_id=from_id, to_claim_id=to_id,
            relationship_type=RelationshipTypeEnum.SIMILAR_TO, score=0.75,
        ))
        await db.flush()  # the first insert must succeed

        db.add(ClaimRelationship(
            from_claim_id=from_id, to_claim_id=to_id,
            relationship_type=RelationshipTypeEnum.SIMILAR_TO, score=0.76,
        ))
        with pytest.raises(IntegrityError):
            await db.flush()

        await db.rollback()  # leave the database exactly as it was found


@pytest.mark.asyncio
async def test_article_embedding_is_in_the_migrated_schema() -> None:
    """Search and corpus stats must work before any optional backfill runs."""
    from app.db.postgres import async_session_maker

    async with async_session_maker() as db:
        row = (
            await db.execute(text(
                "SELECT format_type(a.atttypid, a.atttypmod) "
                "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
                "WHERE c.relname = 'articles' AND a.attname = 'embedding' "
                "AND NOT a.attisdropped"
            ))
        ).scalar_one_or_none()
    assert row == "vector(768)", "article search requires the migrated embedding column"


@pytest.mark.asyncio
async def test_corpus_endpoints_on_migrated_postgres() -> None:
    """Fresh-schema SQL regressions cannot be hidden by frontend demo data."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        stats = await client.get("/api/v1/corpus/stats")
        assert stats.status_code == 200, stats.text
        assert {"total", "embedded", "nlp_counts"} <= set(stats.json())
        articles = await client.get("/api/v1/corpus/articles", params={"limit": 5})
        assert articles.status_code == 200, articles.text
        assert {"total", "items"} <= set(articles.json())


@pytest.mark.asyncio
async def test_mutation_chains_endpoint_runs_on_pgvector() -> None:
    """Regression: /verdicts/game/mutations ordered its pair query by
    ``claims.created_at`` — a column the table never had — so the Mutation DNA
    panel 500'd on real Postgres (asyncpg UndefinedColumnError) while the
    SQLite unit suite stayed green: the ``<=>`` pair query never executes
    there. Assertions are shape-only so they hold on an empty corpus too."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(
            "/api/v1/verdicts/game/mutations",
            params={"min_versions": 2, "limit": 5},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "chains" in body
    for chain in body["chains"]:
        assert {"chain_id", "size", "distinct_outlets", "versions"} <= set(chain)
        for version in chain["versions"]:
            assert {"id", "text", "domain", "published_at"} <= set(version)
