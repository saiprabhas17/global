"""Typed evidence round-trip, parent uniqueness, safe migration and projection."""

import importlib.util
import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.models.article import Article
from app.models.claim import Claim
from app.models.evolution import ClaimRelationship, RelationshipTypeEnum
from app.models.source import Source, SourceTypeEnum
from app.services.evolution.neo4j_writer import Neo4jWriter
from app.services.evolution.text_changes import analyze_text_change

NOW = datetime(2026, 9, 20, tzinfo=UTC)


def _database_uuid(n) -> uuid.UUID:
    # SQLite gives native UUID columns NUMERIC affinity. An all-digit UUID
    # would round-trip as an integer instead of text; use deterministic hex.
    return uuid.UUID(f"abcdef00-0000-0000-0000-{n:012x}")


def _migration() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "gmee01_typed_mutations.py"
    )
    spec = importlib.util.spec_from_file_location("typed_mutations_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_chain_and_upgrade_do_not_silently_remove_historical_edges():
    migration = _migration()
    migration.op = MagicMock()
    migration.upgrade()
    assert migration.revision == "gmee01"
    assert migration.down_revision == "b7d2f4a8c1e9"
    sql = migration.op.execute.call_args.args[0]
    assert "HAVING count(*) > 1" in sql
    assert "RAISE EXCEPTION" in sql
    assert "No historical relationships were removed" in sql
    assert not re.search(r"\b(DELETE|UPDATE|TRUNCATE)\b", sql, re.IGNORECASE)
    column = migration.op.add_column.call_args.args[1]
    assert column.name == "mutation_evidence"
    assert column.nullable is True  # Legacy evidence is unknown, not fabricated.
    index = migration.op.create_index.call_args
    assert index.args == (
        "uq_claim_relationships_evolved_child",
        "claim_relationships",
        ["from_claim_id"],
    )
    assert index.kwargs["unique"] is True
    assert str(index.kwargs["postgresql_where"]) == "relationship_type = 'EVOLVED_FROM'"


def test_migration_conflict_failure_prevents_schema_changes():
    migration = _migration()
    migration.op = MagicMock()
    migration.op.execute.side_effect = RuntimeError(
        "historical claims have multiple parents"
    )
    with pytest.raises(RuntimeError, match="multiple parents"):
        migration.upgrade()
    migration.op.add_column.assert_not_called()
    migration.op.create_index.assert_not_called()


def test_migration_downgrade_removes_only_its_schema_objects():
    migration = _migration()
    migration.op = MagicMock()
    migration.downgrade()
    migration.op.drop_index.assert_called_once_with(
        "uq_claim_relationships_evolved_child", table_name="claim_relationships"
    )
    migration.op.drop_column.assert_called_once_with(
        "claim_relationships", "mutation_evidence"
    )
    migration.op.execute.assert_not_called()


@pytest.mark.asyncio
async def test_evidence_json_round_trip_and_database_single_parent_invariant(
    db_session,
):
    source = Source(
        id=_database_uuid(1),
        name="Test",
        type=SourceTypeEnum.rss,
        url_or_identifier="https://example.com/rss",
    )
    articles = [
        Article(
            id=_database_uuid(100 + i),
            source_id=source.id,
            title=f"Article {i}",
            url=f"https://example.com/{i}",
            content_hash=f"hash-{i}",
            published_at=NOW + timedelta(days=i),
        )
        for i in (1, 2, 3)
    ]
    claims = [
        Claim(
            id=_database_uuid(200 + i),
            article_id=article.id,
            claim_text=f"{10 + i} people",
        )
        for i, article in enumerate(articles, 1)
    ]
    db_session.add(source)
    db_session.add_all(articles)
    await db_session.flush()
    db_session.add_all(claims)
    await db_session.flush()
    analysis = analyze_text_change(
        claims[0].claim_text,
        claims[2].claim_text,
        older_timestamp=articles[0].published_at,
        newer_timestamp=articles[2].published_at,
    )
    edge = ClaimRelationship(
        id=_database_uuid(301),
        from_claim_id=claims[2].id,
        to_claim_id=claims[0].id,
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
        score=0.95,
        mutation_evidence=analysis,
    )
    similar = ClaimRelationship(
        id=_database_uuid(302),
        from_claim_id=claims[2].id,
        to_claim_id=claims[1].id,
        relationship_type=RelationshipTypeEnum.SIMILAR_TO,
        score=0.9,
    )
    db_session.add_all([edge, similar])
    await db_session.commit()
    edge_id, child_id, second_parent_id = edge.id, claims[2].id, claims[1].id
    db_session.expire_all()
    loaded = (
        await db_session.execute(
            select(ClaimRelationship).where(ClaimRelationship.id == edge_id)
        )
    ).scalar_one()
    assert loaded.mutation_evidence == analysis
    assert loaded.mutation_evidence["observed_propagation"] is False
    db_session.add(
        ClaimRelationship(
            id=_database_uuid(303),
            from_claim_id=child_id,
            to_claim_id=second_parent_id,
            relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
            score=0.99,
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()
    count = await db_session.scalar(
        select(func.count(ClaimRelationship.id)).where(
            ClaimRelationship.from_claim_id == child_id
        )
    )
    assert count == 2  # The original parent + SIMILAR_TO survive the failed insert.


@pytest.mark.asyncio
async def test_neo4j_projects_actual_edge_evidence_as_json_and_queryable_types():
    parent, child = uuid.UUID(int=1), uuid.UUID(int=2)
    analysis = analyze_text_change(
        "10 people",
        "12 people",
        older_timestamp=NOW,
        newer_timestamp=NOW + timedelta(days=1),
    )
    edge = ClaimRelationship(
        id=uuid.UUID(int=3),
        from_claim_id=child,
        to_claim_id=parent,
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
        score=0.95,
        created_at=NOW,
        mutation_evidence=analysis,
    )
    session = AsyncMock()
    await Neo4jWriter()._merge_relationships(session, [edge])
    query = session.run.await_args.args[0]
    row = session.run.await_args.kwargs["rows"][0]
    assert "r.mutation_evidence = row.mutation_evidence" in query
    assert "r.observed_propagation = row.observed_propagation" in query
    assert row["from_id"] == str(child)
    assert row["to_id"] == str(parent)
    assert row["child_claim_id"] == str(child)
    assert row["parent_claim_id"] == str(parent)
    assert row["edge_id"] == str(edge.id)
    assert json.loads(row["mutation_evidence"]) == analysis
    assert row["mutation_types"] == analysis["mutation_types"]
    assert row["algorithm_version"] == analysis["algorithm_version"]
    assert row["observed_propagation"] is False


@pytest.mark.asyncio
async def test_neo4j_claim_projection_keeps_article_contract_and_publication_timestamp():
    claim = Claim(
        id=uuid.UUID(int=1),
        article_id=uuid.UUID(int=101),
        claim_text="10 people",
        extracted_at=NOW + timedelta(days=1),
        entities=[],
    )
    article = Article(
        id=claim.article_id, source_id=uuid.UUID(int=201), published_at=NOW
    )
    session = AsyncMock()
    await Neo4jWriter()._merge_claims_entities_sources(
        session, [claim], {article.id: article}, {}
    )
    claim_call = next(
        call for call in session.run.await_args_list if "MERGE (c:Claim" in call.args[0]
    )
    row = claim_call.kwargs["rows"][0]
    assert row["article_id"] == str(article.id)
    assert row["published_at"] == NOW.isoformat()
    assert row["published_at"] != row["extracted_at"]


@pytest.mark.asyncio
async def test_noise_child_projection_prunes_old_parent_even_without_new_edges():
    session = AsyncMock()
    await Neo4jWriter()._prune_stale_evolved(session, {str(uuid.UUID(int=2))}, [])
    assert "DELETE r" in session.run.await_args.args[0]
    assert session.run.await_args.kwargs["claim_ids"] == [str(uuid.UUID(int=2))]
    assert session.run.await_args.kwargs["keep"] == []
