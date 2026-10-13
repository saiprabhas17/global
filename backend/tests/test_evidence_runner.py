"""Batch runner regression tests, without live storage or model inference."""

import json
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.services.verdict.engine import METHOD_VERSION, SCORE_KIND, NLIUnavailableError
from scripts import run_verdicts as runner
from tests.test_evidence_api import response_payload
from tests.test_evidence_check import QUERY, VECTOR, fake_db


def stored_claim(embedding=VECTOR) -> dict[str, Any]:
    return {
        "id": str(UUID(int=10)),
        "article_id": str(UUID(int=1010)),
        "claim_text": QUERY,
        "embedding": embedding,
        "outlet": "own.example",
        "entities": [{"entity_type": "ORG"}] * 5,
        "canonical_article_id": None,
        "content_hash": "own-content",
    }


@pytest.mark.asyncio
async def test_runner_uses_true_passage_stance_and_neutral_default_prior(monkeypatch):
    payload = response_payload()
    payload["evidence"][0]["stance"] = "neutral"
    payload["evidence"][0]["similarity"] = 1.0
    payload["assessment"] = "INSUFFICIENT_EVIDENCE"
    check = AsyncMock(return_value=payload)
    monkeypatch.setattr(runner, "check_claim_with_embedding", check)
    result = await runner.score_claim(fake_db([]), stored_claim())
    assert result["band"] == "UNRESOLVED"
    metadata = json.loads(result["evidence"])
    assert metadata["score_kind"] == SCORE_KIND
    assert metadata["method_version"] == METHOD_VERSION
    assert metadata["source_track_record"]["value"] == 0.5
    assert (
        metadata["source_track_record"]["prior_source"] == "neutral_no_independent_data"
    )
    assert "own labels" in metadata["source_prior_disclosure"]
    assert (
        metadata["checked_neighbors"][0]["passage"] == payload["evidence"][0]["passage"]
    )
    assert metadata["checked_neighbors"][0]["stance"] == "neutral"
    assert check.call_args.kwargs["exclude_claim_id"] == stored_claim()["id"]
    assert check.call_args.kwargs["own_domain"] == "own.example"


@pytest.mark.asyncio
async def test_runner_missing_embedding_records_explicit_abstention(monkeypatch):
    check = AsyncMock(side_effect=AssertionError("No vector means no retrieval"))
    monkeypatch.setattr(runner, "check_claim_with_embedding", check)
    result = await runner.score_claim(fake_db([]), stored_claim(embedding=None))
    metadata = json.loads(result["evidence"])
    assert result["band"] not in (
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "WEAKLY_CORROBORATED",
    )
    assert metadata["corpus_assessment"] == "INSUFFICIENT_EVIDENCE"
    assert metadata["checked_neighbors"] == []
    assert any("retrieval abstained" in warning for warning in metadata["warnings"])
    check.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_independent_prior_requires_provenance_and_records_supplier(
    monkeypatch,
):
    monkeypatch.setattr(
        runner, "check_claim_with_embedding", AsyncMock(return_value=response_payload())
    )
    result = await runner.score_claim(
        fake_db([]),
        stored_claim(),
        {
            "own.example": {
                "supported": 50,
                "disputed": 0,
                "independently_validated": True,
                "provenance": "Independent human-reviewed labels",
            },
        },
    )
    metadata = json.loads(result["evidence"])
    assert (
        metadata["source_track_record"]["provenance"]
        == "Independent human-reviewed labels"
    )
    assert "not been audited" in metadata["source_prior_disclosure"]


def test_no_outlet_prior_file_means_neutral_no_database_query():
    assert runner.load_outlet_priors() == {}


@pytest.mark.parametrize(
    "stats",
    [
        {"supported": 10, "disputed": 0},
        {"supported": 10, "disputed": 0, "independently_validated": True},
        {
            "supported": -1,
            "disputed": 0,
            "independently_validated": True,
            "provenance": "reviewed",
        },
        {
            "supported": True,
            "disputed": 0,
            "independently_validated": True,
            "provenance": "reviewed",
        },
    ],
)
def test_unproven_or_invalid_prior_file_is_rejected(tmp_path, stats):
    path = tmp_path / "priors.json"
    path.write_text(json.dumps({"outlet.example": stats}), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        runner.load_outlet_priors(path)


def test_independent_prior_file_is_normalized(tmp_path):
    path = tmp_path / "priors.json"
    stats = {
        "supported": 10,
        "disputed": 2,
        "independently_validated": True,
        "provenance": "reviewed",
    }
    path.write_text(json.dumps({"WWW.OUTLET.EXAMPLE": stats}), encoding="utf-8")
    assert runner.load_outlet_priors(path) == {"outlet.example": stats}


@pytest.mark.asyncio
@pytest.mark.parametrize("rescore", [False, True])
async def test_persist_preserves_scored_records_unless_explicit_opt_in(rescore):
    db = fake_db([])
    db.execute.return_value.scalar_one_or_none.return_value = UUID(int=1)
    stored = {
        "band": "UNRESOLVED",
        "probability": 0.5,
        "rationale": "uncalibrated",
        "evidence": "{}",
    }
    assert await runner.persist(db, str(UUID(int=1)), stored, rescore=rescore) is True
    statement, params = db.execute.call_args.args
    assert ("AND verdict IS NULL" in str(statement)) is (not rescore)
    assert params["evidence"] == "{}"
    db.commit.assert_not_awaited()


def test_rescoring_is_never_default():
    assert runner.argument_parser().parse_args([]).rescore is False
    assert runner.argument_parser().parse_args(["--rescore"]).rescore is True


@pytest.mark.asyncio
async def test_main_scores_one_pass_without_circular_outlet_history(monkeypatch):
    claims = [stored_claim(), {**stored_claim(), "id": str(UUID(int=11))}]
    load = AsyncMock(return_value=claims)
    score = AsyncMock(return_value={})
    persist = AsyncMock(return_value=True)
    db = fake_db([])
    db.execute.return_value.all.return_value = [("UNRESOLVED", 2)]
    manager = AsyncMock()
    manager.__aenter__.return_value = db
    monkeypatch.setattr(runner, "async_session_maker", lambda: manager)
    monkeypatch.setattr(runner, "load_claims", load)
    monkeypatch.setattr(runner, "score_claim", score)
    monkeypatch.setattr(runner, "persist", persist)
    await runner.main([])
    load.assert_awaited_once_with(100, rescore=False)
    assert score.await_count == 2
    assert all(call.args[2] == {} for call in score.call_args_list)
    assert persist.await_count == 2
    assert all(call.kwargs["rescore"] is False for call in persist.call_args_list)
    assert db.execute.await_count == 1  # Final distribution only, not prior learning.


@pytest.mark.asyncio
async def test_unavailable_model_stops_batch_without_persisting_fake_score(monkeypatch):
    db = fake_db([])
    manager = AsyncMock()
    manager.__aenter__.return_value = db
    persist = AsyncMock()
    monkeypatch.setattr(runner, "async_session_maker", lambda: manager)
    monkeypatch.setattr(runner, "load_claims", AsyncMock(return_value=[stored_claim()]))
    monkeypatch.setattr(
        runner,
        "score_claim",
        AsyncMock(side_effect=NLIUnavailableError("Pre-cache NLI_MODEL")),
    )
    monkeypatch.setattr(runner, "persist", persist)
    with pytest.raises(SystemExit) as exc:
        await runner.main([])
    assert exc.value.code == 1
    persist.assert_not_awaited()
    db.rollback.assert_awaited_once()
    db.commit.assert_not_awaited()
