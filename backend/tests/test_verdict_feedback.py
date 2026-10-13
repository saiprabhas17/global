"""Anonymous feedback is unverified; database writes and client identity are mocked."""

import hashlib
from typing import Any, cast
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import get_db_session
from app.core import ops_security
from app.main import app


@pytest.fixture
async def feedback_db(async_client):
    db = Mock()
    # The awaited result must be a synchronous Mock: children of an AsyncMock
    # are async, so ``scalar_one_or_none()`` would return a truthy coroutine.
    result = Mock()
    result.scalar_one_or_none.return_value = UUID(int=1)
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    previous = app.dependency_overrides[get_db_session]
    app.dependency_overrides[get_db_session] = lambda: db
    yield db
    app.dependency_overrides[get_db_session] = previous


@pytest.mark.asyncio
async def test_engine_config_is_complete():
    from app.services.verdict.engine import engine_config

    cfg = engine_config()
    assert set(cfg["weights"]) == {
        "corroboration",
        "contradiction",
        "source_track_record",
        "entity_grounding",
        "language",
    }
    assert abs(sum(cfg["weights"].values()) - 1.0) < 1e-9
    assert cfg["bands"][0]["band"] == "SUPPORTED"
    assert cfg["near_similarity_window"]["max"] >= cfg["near_similarity_window"]["min"]
    assert cfg["score_kind"] == "uncalibrated_heuristic"
    assert cfg["method_version"]


@pytest.mark.asyncio
async def test_feedback_rejects_non_json(async_client):
    response = await async_client.post("/api/v1/verdicts/feedback", content=b"not json")
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_summary_endpoint_shape_and_non_gold_disclaimer(
    async_client, db_session, monkeypatch
):
    from tests.conftest import TestingSessionLocal

    monkeypatch.setattr("app.api.v1.verdicts.async_session_maker", TestingSessionLocal)
    response = await async_client.get("/api/v1/verdicts/summary")
    assert response.status_code == 200
    data = response.json()
    assert (
        "engine_config" in data and "distribution" in data and "human_feedback" in data
    )
    feedback = data["human_feedback"]
    for key in (
        "total_votes",
        "agree",
        "disagree",
        "agreement_rate",
        "confirmed_disagreements",
    ):
        assert key in feedback
    assert feedback["is_ground_truth"] is False
    assert feedback["feedback_kind"] == "anonymous_unverified"
    assert "not authoritative gold" in feedback["warning"]
    assert "reported alternate band only" in feedback["warning"]


@pytest.mark.asyncio
async def test_feedback_accepts_weakly_corroborated_and_disclaims_authority(
    async_client, feedback_db, monkeypatch
):
    monkeypatch.setattr(
        ops_security, "client_ip", lambda request: "203.0.113.10", raising=False
    )
    response = await async_client.post(
        "/api/v1/verdicts/feedback",
        json={
            "claim_id": str(UUID(int=1)),
            "vote": "DISAGREE",
            "corrected_verdict": "WEAKLY_CORROBORATED",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["is_ground_truth"] is False
    assert data["feedback_kind"] == "anonymous_unverified"
    assert any("not authoritative gold" in warning for warning in data["warnings"])
    statement = feedback_db.execute.call_args_list[-1].args[0]
    assert statement.compile().params["corrected_verdict"] == "WEAKLY_CORROBORATED"
    feedback_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_duplicate_vote_identity_uses_canonical_ip_not_ua_or_untrusted_headers(
    async_client, feedback_db, monkeypatch
):
    resolver = Mock(return_value="203.0.113.10")
    monkeypatch.setattr(ops_security, "client_ip", resolver, raising=False)
    for index, vote in enumerate(("AGREE", "DISAGREE")):
        response = await async_client.post(
            "/api/v1/verdicts/feedback",
            json={
                "claim_id": str(UUID(int=1)),
                "vote": vote,
            },
            headers={
                "x-forwarded-for": f"198.51.100.{index + 1}",
                "user-agent": f"agent-{index}",
            },
        )
        assert response.status_code == 201
    statements = [call.args[0] for call in feedback_db.execute.call_args_list[1::2]]
    hashes = [
        statement.compile(dialect=cast(Any, postgresql.dialect)()).params["client_hash"]
        for statement in statements
    ]
    expected = hashlib.sha256(b"203.0.113.10").hexdigest()
    assert hashes == [expected, expected]
    assert resolver.call_count == 2
    assert all(
        "ON CONFLICT ON CONSTRAINT uq_verdict_feedback_claim_client DO UPDATE"
        in str(statement.compile(dialect=cast(Any, postgresql.dialect)()))
        for statement in statements
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"claim_id": "not-a-valid-uuid-but-32-characters", "vote": "AGREE"},
        {"claim_id": str(UUID(int=1)), "vote": "MAYBE"},
        {"claim_id": str(UUID(int=1)), "vote": "DISAGREE", "corrected_verdict": "FAKE"},
        {"claim_id": str(UUID(int=1)), "vote": "AGREE", "comment": "x" * 1001},
    ],
)
async def test_invalid_feedback_is_422_before_database_work(
    async_client, feedback_db, body
):
    response = await async_client.post("/api/v1/verdicts/feedback", json=body)
    assert response.status_code == 422
    feedback_db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_feedback_claim_is_404(async_client, feedback_db):
    feedback_db.execute.return_value.scalar_one_or_none.return_value = None
    response = await async_client.post(
        "/api/v1/verdicts/feedback",
        json={"claim_id": str(UUID(int=1)), "vote": "AGREE"},
    )
    assert response.status_code == 404
    assert feedback_db.execute.await_count == 1
    feedback_db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_feedback_database_failure_rolls_back_and_is_503(
    async_client, feedback_db
):
    feedback_db.execute.side_effect = SQLAlchemyError("private database error")
    response = await async_client.post(
        "/api/v1/verdicts/feedback",
        json={"claim_id": str(UUID(int=1)), "vote": "AGREE"},
    )
    assert response.status_code == 503
    assert "private" not in response.text
    feedback_db.rollback.assert_awaited_once()
    feedback_db.commit.assert_not_awaited()
