import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import get_settings
from app.models.article import Article
from app.models.claim import Claim
from app.models.evolution import (
    ClaimClusterAssignment,
    ClaimClusterRun,
    ClaimRelationship,
    RelationshipTypeEnum,
)
from app.models.source import Source
from app.services.evolution.cluster_service import CLUSTERING_ALGORITHM_VERSION
from app.services.evolution.mutation_detector import MutationDetector
from app.services.evolution.neo4j_writer import Neo4jWriter
from app.services.evolution.orchestrator import EvolutionOrchestrator


def _result(rows: list[Any]) -> MagicMock:
    """Fake SQLAlchemy Result whose .scalars().all() returns ``rows``."""
    res = MagicMock()
    res.scalars.return_value.all.return_value = rows
    return res


class _MutationFixture(NamedTuple):
    c0: Claim
    c1: Claim
    c2: Claim
    c3: Claim
    assignments: list[ClaimClusterAssignment]
    claims_by_id: dict[uuid.UUID, Claim]
    articles_by_id: dict[uuid.UUID, Article]


def _mutation_fixtures() -> _MutationFixture:
    """One topic, four temporally ordered claims (see the threshold test).

    c1 vs c0 ≈ 0.994 and c2 vs c1 ≈ 0.861 clear the 0.85 EVOLVED_FROM bar;
    c2 vs c0 ≈ 0.80 lands in the SIMILAR_TO window; c3 matches nothing.
    """
    t0 = datetime(2026, 9, 20, tzinfo=UTC)
    t1, t2, t3 = [t0 + timedelta(days=i) for i in (1, 2, 3)]
    art0, art1, art2, art3 = [
        Article(id=uuid.uuid4(), published_at=t) for t in (t0, t1, t2, t3)
    ]
    # Meaningful changes in different, strictly time-ordered articles are
    # required. The previous fixture accidentally asserted same-article and
    # equal-time claims could establish a mutation parent.
    c0 = Claim(id=uuid.uuid4(), article_id=art0.id, claim_text="Workers found 12 survivors.", extracted_at=t0, embedding=[1.0, 0.0, 0.0])
    c1 = Claim(id=uuid.uuid4(), article_id=art1.id, claim_text="Workers found 18 survivors.", extracted_at=t1, embedding=[0.9, 0.1, 0.0])
    c2 = Claim(id=uuid.uuid4(), article_id=art2.id, claim_text="Workers found 30 survivors.", extracted_at=t2, embedding=[0.8, 0.6, 0.0])
    c3 = Claim(id=uuid.uuid4(), article_id=art3.id, claim_text="A different event occurred.", extracted_at=t3, embedding=[0.0, 1.0, 0.0])

    assignments = [
        ClaimClusterAssignment(claim_id=c0.id, topic_id=1),
        ClaimClusterAssignment(claim_id=c1.id, topic_id=1),
        ClaimClusterAssignment(claim_id=c2.id, topic_id=1),
        ClaimClusterAssignment(claim_id=c3.id, topic_id=1),
    ]
    claims_by_id = {c.id: c for c in (c0, c1, c2, c3)}
    articles_by_id = {a.id: a for a in (art0, art1, art2, art3)}
    return _MutationFixture(
        c0, c1, c2, c3, assignments, claims_by_id, articles_by_id
    )


@pytest.mark.asyncio
async def test_orchestrator_guards(monkeypatch):
    settings = get_settings()
    settings.MIN_CORPUS_SIZE_FOR_CLUSTERING = 30
    settings.MIN_NEW_CLAIMS_TO_RECLUSTER = 10

    orchestrator = EvolutionOrchestrator()
    projection = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(orchestrator.neo4j_writer, "sync_persisted_graph", projection)

    # 1. Test small corpus guard
    mock_db = AsyncMock()
    # mock_db.scalar returns 20 for total_claims (less than 30)
    mock_db.scalar.return_value = 20

    res = await orchestrator.run_evolution_cycle(mock_db)
    assert res["status"] == "insufficient_data"
    assert res["claims_available"] == 20
    projection.assert_awaited_once_with(mock_db)

    # 2. Test debounce guard
    mock_db.scalar.side_effect = [
        40,  # total claims > 30 (passes corpus check)
        ClaimClusterRun(
                    claims_in_corpus=35, run_at=datetime.now(UTC),
                    algorithm_params={"algorithm_version": CLUSTERING_ALGORITHM_VERSION},
                )  # current algorithm: debounce is safe
    ]
    # new claims = 40 - 35 = 5 (less than 10 threshold)
    res = await orchestrator.run_evolution_cycle(mock_db)
    assert res["status"] == "debounced"
    assert res["new_claims"] == 5
    # 3. Test force bypass
    mock_db.scalar.side_effect = [
        40,  # total claims > 30
    ]
    exec_result = MagicMock()
    exec_result.scalars.return_value.all.return_value = []
    art_result = MagicMock()
    art_result.scalars.return_value.all.return_value = []
    src_result = MagicMock()
    src_result.scalars.return_value.all.return_value = []
    # 4th execute: orchestrator fetches cluster assignments with an explicit
    # awaited SELECT (never the lazy collection — that raises MissingGreenlet).
    assign_result = MagicMock()
    assign_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(
        side_effect=[exec_result, art_result, src_result, assign_result]
    )
    with patch.object(orchestrator, 'cluster_service') as mock_cluster, \
         patch.object(orchestrator, 'mutation_detector') as mock_mutation, \
         patch.object(orchestrator, 'neo4j_writer') as mock_neo4j:
        
        mock_cluster_run = MagicMock()
        mock_cluster_run.assignments = []
        # The orchestrator awaits these calls, so their mocks must be AsyncMock.
        mock_cluster.run_clustering = AsyncMock(return_value=mock_cluster_run)
        mock_mutation.run_mutation_detection = AsyncMock(return_value=[])
        mock_neo4j.sync_to_graph = AsyncMock(return_value=True)

        res = await orchestrator.run_evolution_cycle(mock_db, force=True)
        assert res["status"] == "success"
        assert res["corpus_size"] == 40


@pytest.mark.asyncio
async def test_mutation_detector_logic():
    settings = get_settings()
    settings.EVOLUTION_SIMILARITY_THRESHOLD = 0.85
    settings.SIMILAR_TO_THRESHOLD = 0.75

    detector = MutationDetector()
    mock_db = AsyncMock()
    # No persisted edges yet: the detector pre-loads existing rows to upsert.
    mock_db.execute = AsyncMock(return_value=_result([]))
    mock_db.add_all = MagicMock()
    fx = _mutation_fixtures()
    c0, c1, c2 = fx.c0, fx.c1, fx.c2
    relationships = await detector.run_mutation_detection(
        mock_db, fx.assignments, fx.claims_by_id, fx.articles_by_id,
    )
    
    # Semantics: each claim links EVOLVED_FROM its single best predecessor
    # (cos-sim >= EVOLUTION_SIMILARITY_THRESHOLD). With this data:
    #   c1 vs c0: ~0.994 >= 0.85 -> EVOLVED_FROM (c1 -> c0)
    #   c2 vs c1: ~0.861 >= 0.85 -> EVOLVED_FROM (c2 -> c1)   [mutation chain]
    #   c2 vs c0: 0.80 in [0.75, 0.85) -> SIMILAR_TO (c2 -> c0)
    #   c3: nothing above thresholds
    
    evo_rels = [r for r in relationships if r.relationship_type == RelationshipTypeEnum.EVOLVED_FROM]
    sim_rels = [r for r in relationships if r.relationship_type == RelationshipTypeEnum.SIMILAR_TO]
    
    assert len(evo_rels) == 2
    evo_pairs = {(r.from_claim_id, r.to_claim_id) for r in evo_rels}
    assert (c1.id, c0.id) in evo_pairs
    assert (c2.id, c1.id) in evo_pairs
    
    for edge in evo_rels:
        child = fx.claims_by_id[edge.from_claim_id]
        parent = fx.claims_by_id[edge.to_claim_id]
        assert child.article_id != parent.article_id
        parent_time = fx.articles_by_id[parent.article_id].published_at
        child_time = fx.articles_by_id[child.article_id].published_at
        assert parent_time is not None and child_time is not None
        assert parent_time < child_time
        assert edge.mutation_evidence is not None
        assert edge.mutation_evidence["meaningful_change"] is True
        assert edge.mutation_evidence["observed_propagation"] is False
        assert "NUMERIC_DRIFT" in edge.mutation_evidence["mutation_types"]
    assert evo_rels[0].from_claim_id != c0.id
    assert sim_rels[0].from_claim_id == c2.id
    assert sim_rels[0].to_claim_id == c0.id


@pytest.mark.asyncio
async def test_neo4j_writer_idempotency():
    # Use a mock driver to check if MERGE is called the expected number of times
    # In reality, testing Neo4j idempotency requires a live Neo4j instance. 
    # Since we can't guarantee a clean Neo4j container in CI without wiping the main one,
    # we mock the driver's session to ensure MERGE statements are executed, 
    # and we rely on Cypher's MERGE semantics for actual idempotency.
    writer = Neo4jWriter()
    
    with patch("app.services.evolution.neo4j_writer.neo4j_client.get_driver") as mock_get_driver:
        mock_driver = AsyncMock()
        mock_get_driver.return_value = mock_driver
        mock_session = AsyncMock()
        async def execute_write(callback, *args):
            return await callback(mock_session, *args)
        mock_session.execute_write.side_effect = execute_write
        # driver.session() is a SYNC factory returning an async context manager;
        # on a plain AsyncMock the call itself would become a coroutine.
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_driver.session = MagicMock(return_value=mock_session_cm)
        
        c0 = Claim(id=uuid.uuid4(), article_id=uuid.uuid4(), claim_text="test", extracted_at=datetime.now(UTC), entities=[])
        art = Article(id=c0.article_id, source_id=uuid.uuid4())
        src = AsyncMock(id=art.source_id, name="Test", type=AsyncMock(value="rss"))
        
        claims = [c0]
        articles = {art.id: art}
        sources = {src.id: cast(Source, src)}
        relationships = [
            ClaimRelationship(
                from_claim_id=c0.id, to_claim_id=c0.id, 
                relationship_type=RelationshipTypeEnum.SIMILAR_TO, score=0.9
            )
        ]
        
        res = await writer.sync_to_graph(claims, articles, sources, relationships)
        assert res is True
        
        # Verify run was called with MERGE - every write batched via UNWIND,
        # never one round-trip per row (4 statements for this slice: sources,
        # claims, PUBLISHED_BY, SIMILAR_TO).
        calls = mock_session.run.call_args_list
        assert len(calls) > 0
        assert len(calls) <= 12
        for call in calls:
            query = call[0][0]
            assert "MERGE" in query or "DELETE" in query
            assert "UNWIND" in query


@pytest.mark.asyncio
async def test_neo4j_writer_prunes_stale_evolved_edges():
    """EVOLVED_FROM edges Postgres no longer holds are deleted in Neo4j.

    Pruning is scoped to claims the run evaluated, and ``keep`` carries only
    the current edge set for those claims - mirroring the Postgres prune so
    the two stores cannot drift.
    """
    writer = Neo4jWriter()

    with patch("app.services.evolution.neo4j_writer.neo4j_client.get_driver") as mock_get_driver:
        mock_driver = AsyncMock()
        mock_get_driver.return_value = mock_driver
        mock_session = AsyncMock()
        async def execute_write(callback, *args):
            return await callback(mock_session, *args)
        mock_session.execute_write.side_effect = execute_write
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_driver.session = MagicMock(return_value=mock_session_cm)

        c0 = Claim(id=uuid.uuid4(), article_id=uuid.uuid4(), claim_text="a", extracted_at=datetime.now(UTC), entities=[])
        c1 = Claim(id=uuid.uuid4(), article_id=c0.article_id, claim_text="b", extracted_at=datetime.now(UTC), entities=[])
        art = Article(id=c0.article_id, source_id=uuid.uuid4())
        src = AsyncMock(id=art.source_id, name="Test", type=AsyncMock(value="rss"))

        kept = ClaimRelationship(
            from_claim_id=c1.id,
            to_claim_id=c0.id,
            relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
            score=0.9,
        )

        res = await writer.sync_to_graph(
            [c0, c1],
            {art.id: art},
            {src.id: cast(Source, src)},
            [kept],
            evaluated_claim_ids={c0.id, c1.id},
        )
        assert res is True

        calls = mock_session.run.call_args_list
        prune_calls = [c for c in calls if "DELETE r" in c[0][0] and "EVOLVED_FROM" in c[0][0]]
        assert len(prune_calls) == 1
        prune_kwargs = prune_calls[0][1]
        assert set(prune_kwargs["claim_ids"]) == {str(c0.id), str(c1.id)}
        assert prune_kwargs["keep"] == [[str(c1.id), str(c0.id)]]


@pytest.mark.asyncio
async def test_orchestrator_resilience_to_neo4j_failure():
    orchestrator = EvolutionOrchestrator()
    mock_db = AsyncMock()
    mock_db.scalar.side_effect = [40]  # total claims > 30

    exec_result = MagicMock()
    exec_result.scalars.return_value.all.return_value = []
    art_result = MagicMock()
    art_result.scalars.return_value.all.return_value = []
    src_result = MagicMock()
    src_result.scalars.return_value.all.return_value = []
    # 4th execute: explicit assignments SELECT (see orchestrator step 4).
    assign_result = MagicMock()
    assign_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(
        side_effect=[exec_result, art_result, src_result, assign_result]
    )

    with patch.object(orchestrator, 'cluster_service') as mock_cluster, \
         patch.object(orchestrator, 'mutation_detector') as mock_mutation, \
         patch.object(orchestrator, 'neo4j_writer') as mock_neo4j:
        
        mock_cluster_run = MagicMock()
        mock_cluster_run.assignments = []
        # The orchestrator awaits these calls, so their mocks must be AsyncMock.
        mock_cluster.run_clustering = AsyncMock(return_value=mock_cluster_run)
        mock_mutation.run_mutation_detection = AsyncMock(return_value=[])

        # Mock Neo4j to fail
        mock_neo4j.sync_to_graph = AsyncMock(return_value=False)

        res = await orchestrator.run_evolution_cycle(mock_db, force=True)
        
        # Assert orchestrator caught the failure and returned false for sync, but overall success
        assert res["status"] == "success"
        assert res["neo4j_sync_success"] is False
        
        # Ensure Postgres transaction was committed
        mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_mutation_detector_upserts_existing_edges():
    """A re-run refreshes stored edges in place instead of duplicating them."""
    settings = get_settings()
    settings.EVOLUTION_SIMILARITY_THRESHOLD = 0.85
    settings.SIMILAR_TO_THRESHOLD = 0.75

    fx = _mutation_fixtures()
    detector = MutationDetector()
    mock_db = AsyncMock()
    mock_db.add_all = MagicMock()

    persisted = ClaimRelationship(
        from_claim_id=fx.c1.id,
        to_claim_id=fx.c0.id,
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
        score=0.10,  # stale score from an earlier run
    )
    mock_db.execute = AsyncMock(return_value=_result([persisted]))

    relationships = await detector.run_mutation_detection(
        mock_db, fx.assignments, fx.claims_by_id, fx.articles_by_id
    )

    added = mock_db.add_all.call_args[0][0]
    added_keys = {
        (r.from_claim_id, r.to_claim_id, r.relationship_type) for r in added
    }
    # the persisted edge is refreshed, never re-inserted
    assert (fx.c1.id, fx.c0.id, RelationshipTypeEnum.EVOLVED_FROM) not in added_keys
    assert {
        (fx.c2.id, fx.c1.id, RelationshipTypeEnum.EVOLVED_FROM),
        (fx.c2.id, fx.c0.id, RelationshipTypeEnum.SIMILAR_TO),
    } <= added_keys
    assert persisted.score > 0.9  # ~0.994 refreshed in place
    mock_db.delete.assert_not_awaited()
    assert len(relationships) == 3


@pytest.mark.asyncio
async def test_mutation_detector_prunes_stale_evolved_from():
    """An EVOLVED_FROM edge the run no longer reproduces is pruned."""
    settings = get_settings()
    settings.EVOLUTION_SIMILARITY_THRESHOLD = 0.85
    settings.SIMILAR_TO_THRESHOLD = 0.75

    fx = _mutation_fixtures()
    detector = MutationDetector()
    mock_db = AsyncMock()
    mock_db.add_all = MagicMock()

    # c2's predecessor is now c1 (≈0.861), so the old c2→c0 EVOLVED_FROM edge
    # is stale; the same pair remains valid as SIMILAR_TO (≈0.80).
    stale = ClaimRelationship(
        from_claim_id=fx.c2.id,
        to_claim_id=fx.c0.id,
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
        score=0.90,
    )
    mock_db.execute = AsyncMock(return_value=_result([stale]))

    await detector.run_mutation_detection(
        mock_db, fx.assignments, fx.claims_by_id, fx.articles_by_id
    )

    deleted = [call.args[0] for call in mock_db.delete.await_args_list]
    assert deleted == [stale]

    added = mock_db.add_all.call_args[0][0]
    assert any(
        r.relationship_type == RelationshipTypeEnum.SIMILAR_TO
        and r.from_claim_id == fx.c2.id
        and r.to_claim_id == fx.c0.id
        for r in added
    )
