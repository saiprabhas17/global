"""Parent semantics and reconciliation without downloaded embeddings/models."""

import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.article import Article
from app.models.claim import Claim
from app.models.evolution import (
    ClaimClusterAssignment,
    ClaimRelationship,
    RelationshipTypeEnum,
)
from app.services.evolution.mutation_detector import MutationDetector, candidate_limits

NOW = datetime(2026, 9, 20, tzinfo=UTC)


def _claim(index, text, published_at, *, embedding=None, article_id=None) -> tuple[Claim, Article]:
    aid = article_id or uuid.UUID(int=1000 + index)
    article = Article(id=aid, published_at=published_at)
    claim = Claim(
        id=uuid.UUID(int=index),
        article_id=aid,
        claim_text=text,
        extracted_at=NOW + timedelta(hours=index),
        embedding=[1.0, 0.0] if embedding is None else embedding,
    )
    return claim, article


def _db(existing=()) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(existing)
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    db.flush = AsyncMock()
    db.delete = AsyncMock()
    return db


async def _detect(
    pairs, *, existing=(), topics=None, candidate_limit=50, max_lag_days=30
) -> tuple[list[ClaimRelationship], MagicMock]:
    detector = MutationDetector()
    settings = SimpleNamespace(
        EVOLUTION_SIMILARITY_THRESHOLD=0.85,
        SIMILAR_TO_THRESHOLD=0.75,
        MUTATION_CANDIDATE_LIMIT=candidate_limit,
        MUTATION_MAX_LAG_DAYS=max_lag_days,
    )
    assignments = [
        ClaimClusterAssignment(claim_id=c.id, topic_id=topics[i] if topics else 1)
        for i, (c, _) in enumerate(pairs)
    ]
    db = _db(existing)
    with patch.object(detector, "settings", settings):
        edges = await detector.run_mutation_detection(
            db,
            assignments,
            {c.id: c for c, _ in pairs},
            {a.id: a for _, a in pairs},
        )
    return edges, db


def _evolved(edges) -> list[ClaimRelationship]:
    return [
        e for e in edges if e.relationship_type == RelationshipTypeEnum.EVOLVED_FROM
    ]


def test_optional_settings_defaults_and_absolute_candidate_cap():
    assert candidate_limits(SimpleNamespace()) == (50, 30)
    assert candidate_limits(
        SimpleNamespace(MUTATION_CANDIDATE_LIMIT=999999, MUTATION_MAX_LAG_DAYS=10)
    ) == (500, 10)


@pytest.mark.parametrize("new_text", ["The count is 10.", " the COUNT is 10! "])
@pytest.mark.asyncio
async def test_no_change_duplicates_are_similarity_not_mutation(new_text):
    older = _claim(1, "The count is 10.", NOW)
    newer = _claim(2, new_text, NOW + timedelta(days=1))
    edges, _ = await _detect([older, newer])
    assert len(edges) == 1
    assert not _evolved(edges)
    assert edges[0].relationship_type == RelationshipTypeEnum.SIMILAR_TO
    assert edges[0].mutation_evidence is not None
    assert edges[0].mutation_evidence["mutation_types"] == ["NEAR_DUPLICATE"]
    assert edges[0].mutation_evidence is not None
    assert edges[0].mutation_evidence["meaningful_change"] is False


@pytest.mark.parametrize(
    "times",
    [
        (NOW, NOW),
        (None, NOW),
        (NOW, None),
        (None, None),
    ],
)
@pytest.mark.asyncio
async def test_equal_or_missing_publication_times_never_establish_parent(times):
    pairs = [
        _claim(1, "10 people arrived.", times[0]),
        _claim(2, "12 people arrived.", times[1]),
    ]
    edges, _ = await _detect(pairs)
    assert not _evolved(edges)
    assert len(edges) == 1
    assert edges[0].relationship_type == RelationshipTypeEnum.SIMILAR_TO
    assert edges[0].mutation_evidence is not None
    assert edges[0].mutation_evidence["parent_eligibility"]["strictly_older"] is False
    # extracted_at is present and strictly ordered, but isn't publication evidence.
    assert pairs[0][0].extracted_at < pairs[1][0].extracted_at


@pytest.mark.asyncio
async def test_same_article_claims_do_not_create_parent():
    pairs = [
        _claim(1, "10 people arrived.", NOW),
        _claim(2, "12 people arrived.", NOW, article_id=uuid.UUID(int=1001)),
    ]
    edges, _ = await _detect(pairs)
    assert not _evolved(edges)
    assert edges[0].mutation_evidence is not None
    assert (
        edges[0].mutation_evidence["parent_eligibility"]["different_article"] is False
    )


@pytest.mark.asyncio
async def test_best_duplicate_does_not_hide_a_meaningful_candidate():
    pairs = [
        _claim(1, "10 people arrived.", NOW, embedding=[1.0, 0.1]),
        _claim(2, "12 people arrived.", NOW + timedelta(days=1)),
        _claim(3, "12 people arrived.", NOW + timedelta(days=2)),
    ]
    edges, _ = await _detect(pairs)
    child_edges = [e for e in edges if e.from_claim_id == pairs[2][0].id]
    assert [(e.from_claim_id, e.to_claim_id) for e in _evolved(child_edges)] == [
        (pairs[2][0].id, pairs[0][0].id)
    ]
    duplicate = next(e for e in child_edges if e.to_claim_id == pairs[1][0].id)
    assert duplicate.relationship_type == RelationshipTypeEnum.SIMILAR_TO
    assert duplicate.score > _evolved(child_edges)[0].score


@pytest.mark.asyncio
async def test_candidate_cap_is_enforced_and_reported_not_claimed_exhaustive():
    pairs = [
        _claim(1, "10 people arrived.", NOW, embedding=[1.0, 0.1]),
        _claim(2, "12 people arrived.", NOW + timedelta(days=1)),
        _claim(3, "12 people arrived.", NOW + timedelta(days=2)),
    ]
    edges, _ = await _detect(pairs, candidate_limit=1)
    child_edges = [e for e in edges if e.from_claim_id == pairs[2][0].id]
    assert len(child_edges) == 1
    assert not _evolved(child_edges)
    assert child_edges[0].mutation_evidence is not None
    assert child_edges[0].mutation_evidence["selection"]["candidate_limit"] == 1
    assert child_edges[0].mutation_evidence is not None
    assert "missed" in " ".join(
        child_edges[0].mutation_evidence["selection"]["limitations"]
    )


@pytest.mark.asyncio
async def test_strict_parent_with_mixed_timezone_conventions_and_evidence():
    pairs = [
        _claim(1, "10 people arrived.", NOW.replace(tzinfo=None)),
        _claim(2, "12 people arrived.", NOW + timedelta(hours=1)),
    ]
    edges, _ = await _detect(pairs)
    assert len(_evolved(edges)) == 1
    evidence = _evolved(edges)[0].mutation_evidence
    assert evidence is not None
    assert evidence["lag_seconds"] == 3600
    assert evidence["parent_claim_id"] == str(pairs[0][0].id)
    assert evidence["child_claim_id"] == str(pairs[1][0].id)
    assert evidence["observed_propagation"] is False
    assert evidence["selection"]["temporal_basis"] == "article.published_at_only"
    assert "NUMERIC_DRIFT" in evidence["mutation_types"]


@pytest.mark.parametrize(("days", "has_parent"), [(30, True), (31, False)])
@pytest.mark.asyncio
async def test_maximum_lag_bounds_parent_candidates(days, has_parent):
    pairs = [
        _claim(1, "10 people arrived.", NOW),
        _claim(2, "12 people arrived.", NOW + timedelta(days=days)),
    ]
    edges, _ = await _detect(pairs)
    assert bool(_evolved(edges)) is has_parent


@pytest.mark.parametrize("topics", [[1, 2], [-1, -1]])
@pytest.mark.asyncio
async def test_stale_parent_reconciled_when_clusters_split_or_become_noise(topics):
    pairs = [
        _claim(1, "10 people arrived.", NOW),
        _claim(2, "12 people arrived.", NOW + timedelta(days=1)),
    ]
    stale = ClaimRelationship(
        from_claim_id=pairs[1][0].id,
        to_claim_id=pairs[0][0].id,
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
        score=0.95,
    )
    edges, db = await _detect(pairs, existing=[stale], topics=topics)
    assert edges == []
    db.delete.assert_awaited_once_with(stale)
    # A target-membership predicate would have missed old cross-cluster parents.
    query = str(db.execute.await_args_list[-1].args[0])
    assert "from_claim_id IN" in query
    assert "to_claim_id IN" not in query


@pytest.mark.asyncio
async def test_singleton_cleans_parent_outside_supplied_corpus_but_not_unrelated_child():
    child = _claim(2, "12 people arrived.", NOW)
    stale = ClaimRelationship(
        from_claim_id=child[0].id,
        to_claim_id=uuid.UUID(int=999),
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
        score=0.95,
    )
    unrelated = ClaimRelationship(
        from_claim_id=uuid.UUID(int=998),
        to_claim_id=uuid.UUID(int=997),
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
        score=0.95,
    )
    edges, db = await _detect([child], existing=[stale, unrelated])
    assert edges == []
    db.delete.assert_awaited_once_with(stale)


@pytest.mark.asyncio
async def test_missing_embedding_and_assignment_still_reconciles_child():
    child, article = _claim(2, "12 people arrived.", NOW)
    child.embedding = None
    stale = ClaimRelationship(
        from_claim_id=child.id,
        to_claim_id=uuid.UUID(int=999),
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
        score=0.95,
    )
    detector, db = MutationDetector(), _db([stale])
    assert (
        await detector.run_mutation_detection(
            db, [], {child.id: child}, {article.id: article}
        )
        == []
    )
    db.delete.assert_awaited_once_with(stale)


@pytest.mark.asyncio
async def test_legacy_multiple_parents_reconciled_before_new_edge_insert():
    pairs = [
        _claim(1, "10 people arrived.", NOW),
        _claim(2, "12 people arrived.", NOW + timedelta(days=1)),
    ]
    stale = [
        ClaimRelationship(
            from_claim_id=pairs[1][0].id,
            to_claim_id=uuid.UUID(int=i),
            relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
            score=0.95,
        )
        for i in (998, 999)
    ]
    edges, db = await _detect(pairs, existing=stale)
    assert len(_evolved(edges)) == 1
    assert {c.args[0] for c in db.delete.await_args_list} == set(stale)
    methods = [call[0] for call in db.mock_calls]
    assert methods.index("flush") < methods.index("add_all")
    lock_query = str(db.execute.await_args_list[0].args[0])
    assert "FOR UPDATE" in lock_query


@pytest.mark.asyncio
async def test_existing_chosen_parent_is_upserted_and_extra_parent_removed():
    pairs = [
        _claim(1, "10 people arrived.", NOW),
        _claim(2, "12 people arrived.", NOW + timedelta(days=1)),
    ]
    kept = ClaimRelationship(
        from_claim_id=pairs[1][0].id,
        to_claim_id=pairs[0][0].id,
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
        score=0.1,
    )
    stale = ClaimRelationship(
        from_claim_id=pairs[1][0].id,
        to_claim_id=uuid.UUID(int=999),
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM,
        score=0.95,
    )
    edges, db = await _detect(pairs, existing=[kept, stale])
    assert _evolved(edges) == [kept]
    assert kept.score == pytest.approx(1.0)
    assert kept.mutation_evidence is not None
    assert kept.mutation_evidence is not None
    assert kept.mutation_evidence["algorithm_version"]
    db.add_all.assert_not_called()
    db.delete.assert_awaited_once_with(stale)


@pytest.mark.asyncio
async def test_assignment_and_input_order_do_not_change_edge_selection():
    pairs = [
        _claim(i, f"{10 + i} people arrived.", NOW + timedelta(days=i))
        for i in range(1, 8)
    ]
    first, _ = await _detect(pairs)
    second, _ = await _detect(list(reversed(pairs)))
    def signature(edges: list[ClaimRelationship]) -> list[tuple[Any, ...]]:
        return [(e.from_claim_id, e.to_claim_id, e.relationship_type, e.mutation_evidence)
                for e in edges]
    assert signature(first) == signature(second)
    assert max(Counter(e.from_claim_id for e in _evolved(first)).values()) == 1


@pytest.mark.asyncio
async def test_nonfinite_embeddings_do_not_break_other_candidates():
    pairs = [
        _claim(1, "10 people arrived.", NOW),
        _claim(
            2,
            "11 people arrived.",
            NOW + timedelta(days=1),
            embedding=[float("nan"), 0],
        ),
        _claim(3, "12 people arrived.", NOW + timedelta(days=2)),
    ]
    edges, _ = await _detect(pairs)
    assert [(e.from_claim_id, e.to_claim_id) for e in _evolved(edges)] == [
        (pairs[2][0].id, pairs[0][0].id)
    ]
