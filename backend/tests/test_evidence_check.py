"""Corpus evidence tests with mocked models and database, never live pgvector."""

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest

from app.services.verdict import evidence
from app.services.verdict.engine import (
    METHOD_VERSION,
    SCORE_KIND,
    NLIUnavailableError,
    StanceResult,
)

NOW = datetime(2025, 6, 1, tzinfo=UTC)
PAST = datetime(2024, 1, 1, tzinfo=UTC)
QUERY = "The council approved the new bridge."
VECTOR = [1.0] + [0.0] * 767


def corpus_row(index=1, **overrides) -> dict[str, Any]:
    row = {
        "claim_id": str(UUID(int=index)),
        "article_id": str(UUID(int=1000 + index)),
        "text": QUERY,
        "url": f"https://outlet{index}.example/news/{index}",
        "title": f"Council report {index}",
        "domain": f"outlet{index}.example",
        "published_at": PAST,
        "collected_at": PAST,
        "extracted_at": PAST,
        "similarity": 0.9,
        "canonical_article_id": None,
        "content_hash": f"hash-{index}",
        "cleaned_content": f"Report {index}. {QUERY} Work starts next year.",
        "content": None,
    }
    row.update(overrides)
    return row


def fake_db(rows) -> Mock:
    result = Mock()
    result.mappings.return_value.all.return_value = rows
    result.mappings.return_value.first.return_value = rows[0] if rows else None
    db = Mock()
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_true_stance_not_similarity_determines_assessment(monkeypatch):
    db = fake_db([corpus_row(similarity=1.0)])
    classify = AsyncMock(return_value=StanceResult("neutral"))
    monkeypatch.setattr(evidence, "classify_stance", classify)
    result = await evidence.check_claim_with_embedding(
        db, QUERY, VECTOR, observed_at=NOW
    )
    assert result["assessment"] == "INSUFFICIENT_EVIDENCE"
    assert result["evidence"][0]["stance"] == "neutral"
    assert result["score_kind"] == SCORE_KIND
    assert result["method_version"] == METHOD_VERSION
    assert "probability" not in result
    assert "confidence" not in result["evidence"][0]
    assert any("Corpus-local coverage" in warning for warning in result["warnings"])
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stances,expected",
    [
        (["entailment"], "SUPPORTED_BY_CORPUS"),
        (["contradiction"], "CONTRADICTED_BY_CORPUS"),
        (["entailment", "contradiction"], "MIXED_EVIDENCE"),
        (["neutral", "neutral"], "INSUFFICIENT_EVIDENCE"),
        ([], "INSUFFICIENT_EVIDENCE"),
    ],
)
async def test_assessment_four_way_contract(monkeypatch, stances, expected):
    db = fake_db([corpus_row(i + 1) for i in range(len(stances))])
    classify = AsyncMock(side_effect=[StanceResult(stance) for stance in stances])
    monkeypatch.setattr(evidence, "classify_stance", classify)
    result = await evidence.check_claim_with_embedding(
        db, QUERY, VECTOR, observed_at=NOW
    )
    assert result["assessment"] == expected
    assert classify.await_count == len(stances)


@pytest.mark.asyncio
async def test_nli_checks_actual_returned_article_passage_not_extracted_claim(
    monkeypatch,
):
    body = "The opposition claimed approval. The council did not approve the new bridge. No work starts."
    row = corpus_row(cleaned_content=body)
    classify = AsyncMock(return_value=StanceResult("contradiction"))
    monkeypatch.setattr(evidence, "classify_stance", classify)
    result = await evidence.check_claim_with_embedding(
        fake_db([row]), QUERY, VECTOR, observed_at=NOW
    )
    item = result["evidence"][0]
    assert item["passage"] in body
    assert "did not approve" in item["passage"]
    assert item["passage_source"] == "cleaned_content"
    classify.assert_awaited_once_with(QUERY, item["passage"])
    assert result["assessment"] == "CONTRADICTED_BY_CORPUS"


def test_passages_are_contiguous_source_substrings_including_context():
    body = "Some background. The council approved the new bridge. Further details."
    candidate = evidence._candidate(corpus_row(cleaned_content=body))
    passage, source = evidence.evidence_passage(candidate, QUERY)
    assert passage in body
    assert QUERY in passage
    assert "Some background." in passage
    assert "Further details." in passage
    assert source == "cleaned_content"


def test_raw_stored_content_is_used_when_cleaned_content_absent():
    body = "Stored raw text: The council approved the new bridge."
    candidate = evidence._candidate(corpus_row(cleaned_content=None, content=body))
    passage, source = evidence.evidence_passage(candidate, QUERY)
    assert passage in body
    assert source == "content"


@pytest.mark.asyncio
async def test_extracted_claim_fallback_is_explicit_not_a_fabricated_quote(monkeypatch):
    row = corpus_row(cleaned_content=None, content=None)
    monkeypatch.setattr(
        evidence, "classify_stance", AsyncMock(return_value=StanceResult("entailment"))
    )
    result = await evidence.check_claim_with_embedding(
        fake_db([row]), QUERY, VECTOR, observed_at=NOW
    )
    item = result["evidence"][0]
    assert item["passage"] == row["text"]
    assert item["passage_source"] == "extracted_claim"
    assert any("not an article quotation" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_historical_mode_excludes_future_and_unknown_publications(monkeypatch):
    cutoff = datetime(2024, 6, 1, tzinfo=UTC)
    rows = [
        corpus_row(1),
        corpus_row(2, published_at=cutoff + timedelta(seconds=1)),
        corpus_row(3, published_at=None),
        corpus_row(4, collected_at=cutoff + timedelta(seconds=1)),
        corpus_row(5, extracted_at=cutoff + timedelta(seconds=1)),
        corpus_row(6, collected_at=None),
        corpus_row(7, published_at=cutoff, collected_at=cutoff, extracted_at=cutoff),
    ]
    db = fake_db(rows)
    classify = AsyncMock(return_value=StanceResult("neutral"))
    monkeypatch.setattr(evidence, "classify_stance", classify)
    result = await evidence.check_claim_with_embedding(
        db, QUERY, VECTOR, as_of=cutoff, observed_at=NOW
    )
    assert {item["claim_id"] for item in result["evidence"]} == {
        rows[0]["claim_id"],
        rows[6]["claim_id"],
    }
    statement, params = db.execute.call_args.args
    sql = str(statement)
    assert "a.published_at <= :cutoff" in sql
    assert "a.collected_at <= :cutoff" in sql
    assert "c.extracted_at <= :cutoff" in sql
    assert params["cutoff"] == cutoff
    assert params["historical"] is True
    assert any("Historical mode" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_current_mode_never_includes_future_but_discloses_unknown_time(
    monkeypatch,
):
    rows = [
        corpus_row(1, published_at=NOW + timedelta(seconds=1)),
        corpus_row(2, published_at=None),
    ]
    monkeypatch.setattr(
        evidence, "classify_stance", AsyncMock(return_value=StanceResult("neutral"))
    )
    result = await evidence.check_claim_with_embedding(
        fake_db(rows), QUERY, VECTOR, observed_at=NOW
    )
    assert [item["claim_id"] for item in result["evidence"]] == [rows[1]["claim_id"]]
    assert any("unknown publication time" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_future_as_of_is_capped_and_naive_as_of_disclosed(monkeypatch):
    db = fake_db([corpus_row(published_at=NOW + timedelta(days=1))])
    monkeypatch.setattr(
        evidence, "classify_stance", AsyncMock(return_value=StanceResult("neutral"))
    )
    result = await evidence.check_claim_with_embedding(
        db,
        QUERY,
        VECTOR,
        as_of=NOW.replace(tzinfo=None) + timedelta(days=2),
        observed_at=NOW,
    )
    assert not result["evidence"]
    assert db.execute.call_args.args[1]["cutoff"] == NOW
    assert any("capped" in warning for warning in result["warnings"])
    assert any("interpreted as UTC" in warning for warning in result["warnings"])


def test_canonical_links_and_exact_content_hashes_group_transitively():
    root_id = str(UUID(int=5000))
    candidates = [
        evidence._candidate(
            corpus_row(1, canonical_article_id=root_id, content_hash="shared")
        ),
        evidence._candidate(corpus_row(2, canonical_article_id=root_id)),
        evidence._candidate(corpus_row(3, content_hash="shared")),
        evidence._candidate(corpus_row(4)),
    ]
    grouped = evidence.group_candidates(candidates)
    assert len({candidate.syndication_group for candidate in grouped[:3]}) == 1
    assert grouped[0].syndication_group == f"canonical:{root_id}"
    assert grouped[3].syndication_group != grouped[0].syndication_group
    assert len(evidence.select_candidates(candidates, 12)) == 2


def test_identical_claim_wording_is_not_itself_a_syndication_group():
    candidates = [evidence._candidate(corpus_row(i)) for i in (1, 2)]
    assert candidates[0].text == candidates[1].text
    assert (
        len({c.syndication_group for c in evidence.group_candidates(candidates)}) == 2
    )


def test_cross_domain_selection_occurs_before_limit_is_filled():
    candidates = [
        evidence._candidate(corpus_row(1, domain="a.example", similarity=0.99)),
        evidence._candidate(corpus_row(2, domain="a.example", similarity=0.98)),
        evidence._candidate(corpus_row(3, domain="a.example", similarity=0.97)),
        evidence._candidate(corpus_row(4, domain="b.example", similarity=0.8)),
        evidence._candidate(corpus_row(5, domain="c.example", similarity=0.7)),
    ]
    assert [c.domain for c in evidence.select_candidates(candidates, 3)] == [
        "a.example",
        "b.example",
        "c.example",
    ]


def test_batch_excludes_own_domain_and_syndicated_own_article():
    own = evidence._candidate(
        corpus_row(1, domain="www.a.example", content_hash="own-hash")
    )
    candidates = [
        replace(own, claim_id=str(UUID(int=2))),
        evidence._candidate(corpus_row(3, domain="b.example", content_hash="own-hash")),
        evidence._candidate(corpus_row(4, domain="c.example")),
    ]
    selected = evidence.select_candidates(
        candidates, 6, own_domain="a.example", own_origin=own
    )
    assert [candidate.domain for candidate in selected] == ["c.example"]


@pytest.mark.asyncio
async def test_candidate_and_nli_work_are_bounded_and_vectors_bound(monkeypatch):
    db = fake_db([corpus_row(i) for i in range(1, 150)])
    classify = AsyncMock(return_value=StanceResult("neutral"))
    monkeypatch.setattr(evidence, "classify_stance", classify)
    result = await evidence.check_claim_with_embedding(
        db, QUERY, VECTOR, limit=12, observed_at=NOW
    )
    assert len(result["evidence"]) == 12
    assert classify.await_count == 12
    statement, params = db.execute.call_args.args
    assert params["candidate_limit"] == evidence.MAX_CANDIDATES
    assert params["content_cap"] == evidence.MAX_CONTENT_CHARS
    assert "CAST(:embedding AS vector)" in str(statement)
    assert "LIMIT :candidate_limit" in str(statement)
    assert json_vector_length(params["embedding"]) == 768
    assert "UPDATE" not in str(statement)


def json_vector_length(value) -> int:
    import json

    return len(json.loads(value))


@pytest.mark.asyncio
async def test_no_eligible_candidates_does_not_load_nli(monkeypatch):
    classify = AsyncMock(side_effect=AssertionError("NLI must not be needed"))
    monkeypatch.setattr(evidence, "classify_stance", classify)
    result = await evidence.check_claim_with_embedding(
        fake_db([corpus_row(similarity=0.1)]), QUERY, VECTOR, observed_at=NOW
    )
    assert result["assessment"] == "INSUFFICIENT_EVIDENCE"
    classify.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_mapping_is_explicit_abstention_warning(monkeypatch):
    monkeypatch.setattr(
        evidence,
        "classify_stance",
        AsyncMock(return_value=StanceResult("neutral", "unknown_label_mapping")),
    )
    result = await evidence.check_claim_with_embedding(
        fake_db([corpus_row()]), QUERY, VECTOR, observed_at=NOW
    )
    assert result["assessment"] == "INSUFFICIENT_EVIDENCE"
    assert any("unknown label mapping" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_unavailable_nli_propagates_not_fake_neutral_or_support(monkeypatch):
    monkeypatch.setattr(
        evidence,
        "classify_stance",
        AsyncMock(side_effect=NLIUnavailableError("Pre-cache local NLI weights")),
    )
    with pytest.raises(NLIUnavailableError, match="Pre-cache"):
        await evidence.check_claim_with_embedding(
            fake_db([corpus_row()]), QUERY, VECTOR, observed_at=NOW
        )


@pytest.mark.asyncio
async def test_embedding_unavailable_uses_disclosed_keyword_retrieval(monkeypatch):
    from app.services.nlp.embedding_service import EmbeddingService

    generate = Mock(side_effect=RuntimeError("not loaded"))
    load = Mock(side_effect=AssertionError("must not load on request"))
    monkeypatch.setattr(EmbeddingService, "generate_embedding", generate)
    monkeypatch.setattr(EmbeddingService, "load_model", load)
    db = fake_db([])
    result = await evidence.check_claim(db, QUERY)
    load.assert_not_called()
    db.execute.assert_awaited_once()
    assert db.execute.call_args.args[1]["embedding"] is None
    assert any("keyword retrieval was used" in item for item in result["warnings"])
    assert result["assessment"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_keyword_only_hit_can_supply_traceable_evidence(monkeypatch):
    body = "  The council approved the new bridge. This is the published text.  "
    row = corpus_row(similarity=None, lexical_score=0.12, cleaned_content=body)
    monkeypatch.setattr(
        evidence, "classify_stance", AsyncMock(return_value=StanceResult("entailment"))
    )
    result = await evidence.check_claim_with_embedding(
        fake_db([row]), QUERY, None, observed_at=NOW
    )
    item = result["evidence"][0]
    assert result["assessment"] == "SUPPORTED_BY_CORPUS"
    assert item["similarity"] is None
    assert item["retrieval_method"] == "keyword"
    assert body[item["passage_start"] : item["passage_end"]] == item["passage"]
    assert item["passage_sha256"] == hashlib.sha256(item["passage"].encode()).hexdigest()
    assert item["collected_at"] == PAST
    assert item["extracted_at"] == PAST


@pytest.mark.asyncio
async def test_hybrid_ranking_preserves_time_filter_in_both_channels(monkeypatch):
    db = fake_db([
        corpus_row(1, similarity=0.99, retrieval_score=0.015),
        corpus_row(2, similarity=0.8, lexical_score=0.5, retrieval_score=0.03),
    ])
    monkeypatch.setattr(
        evidence, "classify_stance", AsyncMock(return_value=StanceResult("neutral"))
    )
    result = await evidence.check_claim_with_embedding(db, QUERY, VECTOR, observed_at=NOW)
    assert [item["claim_id"] for item in result["evidence"]] == [
        str(UUID(int=2)), str(UUID(int=1))
    ]
    assert result["evidence"][0]["retrieval_method"] == "hybrid"
    sql, params = db.execute.call_args.args
    for channel in str(sql).split("), fused AS")[0].split("), keyword_hits AS"):
        assert "c.extracted_at <= :cutoff" in channel
        assert "a.collected_at <= :cutoff" in channel
        assert channel.index("a.published_at <= :cutoff") < channel.index("LIMIT :candidate_limit")
    assert QUERY not in str(sql)
    assert params["lexical_query"] == "the OR council OR approved OR new OR bridge"


@pytest.mark.parametrize(
    "vector", [[0.0] * 768, [1.0] * 3, [float("nan")] * 768, "not a vector"]
)
def test_invalid_or_incompatible_embeddings_are_actionable(vector):
    with pytest.raises(evidence.EmbeddingUnavailableError, match="768-dimensional"):
        evidence.vector_literal(vector)


@pytest.mark.asyncio
async def test_malformed_candidate_is_excluded_not_fabricated(monkeypatch):
    row = corpus_row(similarity=float("nan"))
    monkeypatch.setattr(evidence, "classify_stance", AsyncMock())
    result = await evidence.check_claim_with_embedding(
        fake_db([row]), QUERY, VECTOR, observed_at=NOW
    )
    assert result["assessment"] == "INSUFFICIENT_EVIDENCE"
    assert any("malformed" in warning for warning in result["warnings"])
