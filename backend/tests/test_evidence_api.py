"""Authenticated check and stored-verdict API contracts, all dependencies mocked."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import get_current_user, get_db_session
from app.api.v1 import verdicts
from app.services.verdict.engine import METHOD_VERSION, SCORE_KIND, NLIUnavailableError
from app.services.verdict.evidence import EmbeddingUnavailableError
from tests.test_evidence_check import NOW, QUERY, corpus_row, fake_db


@pytest.fixture
def api_db():
    return fake_db([])


@pytest.fixture
def verdict_app(api_db):
    app = FastAPI()
    app.include_router(verdicts.router, prefix="/api/v1/verdicts")
    app.dependency_overrides[get_db_session] = lambda: api_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=UUID(int=42)
    )
    return app


@pytest.fixture
async def client(verdict_app):
    async with AsyncClient(
        transport=ASGITransport(app=verdict_app), base_url="http://test"
    ) as client:
        yield client


def response_payload() -> dict[str, Any]:
    row = corpus_row()
    return {
        "claim_text": QUERY,
        "assessment": "SUPPORTED_BY_CORPUS",
        "evidence": [
            {
                "claim_id": row["claim_id"],
                "article_id": row["article_id"],
                "text": row["text"],
                "passage": row["cleaned_content"],
                "passage_source": "cleaned_content",
                "url": row["url"],
                "title": row["title"],
                "domain": row["domain"],
                "published_at": row["published_at"],
                "similarity": 0.9,
                "stance": "entailment",
                "syndication_group": "content:hash-1",
            }
        ],
        "warnings": ["Corpus-local coverage only; not independent verification."],
        "score_kind": SCORE_KIND,
        "observed_at": NOW,
        "method_version": METHOD_VERSION,
    }


@pytest.mark.asyncio
async def test_check_requires_authentication(client, verdict_app, api_db, monkeypatch):
    verdict_app.dependency_overrides.pop(get_current_user)
    check = AsyncMock()
    monkeypatch.setattr(verdicts, "check_claim", check)
    response = await client.post("/api/v1/verdicts/check", json={"claim_text": QUERY})
    assert response.status_code in (401, 403)
    check.assert_not_awaited()
    api_db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_has_explicit_response_no_probability_and_uses_injected_db(
    client, api_db, monkeypatch
):
    check = AsyncMock(return_value=response_payload())
    monkeypatch.setattr(verdicts, "check_claim", check)
    response = await client.post(
        "/api/v1/verdicts/check",
        json={
            "claim_text": f"  {QUERY}  ",
            "as_of": "2024-06-01T00:00:00Z",
            "limit": 4,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "claim_text",
        "assessment",
        "evidence",
        "warnings",
        "score_kind",
        "observed_at",
        "method_version",
    }
    assert data["score_kind"] == "uncalibrated_heuristic"
    assert data["assessment"] == "SUPPORTED_BY_CORPUS"
    assert "probability" not in json.dumps(data)
    assert "confidence" not in json.dumps(data)
    assert set(data["evidence"][0]) == {
        "claim_id",
        "article_id",
        "text",
        "passage",
        "passage_source",
        "url",
        "title",
        "domain",
        "published_at",
        "similarity",
        "stance",
        "syndication_group",
        "retrieval_method",
        "collected_at",
        "extracted_at",
        "passage_start",
        "passage_end",
        "passage_sha256",
        "abstention_reason",
    }
    check.assert_awaited_once_with(
        api_db,
        QUERY,
        as_of=datetime(2024, 6, 1, tzinfo=UTC),
        limit=4,
    )
    api_db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_optional_check_fields_have_bounded_defaults(client, api_db, monkeypatch):
    check = AsyncMock(return_value=response_payload())
    monkeypatch.setattr(verdicts, "check_claim", check)
    response = await client.post("/api/v1/verdicts/check", json={"claim_text": QUERY})
    assert response.status_code == 200
    check.assert_awaited_once_with(api_db, QUERY, as_of=None, limit=6)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"claim_text": "short"},
        {"claim_text": " " * 30},
        {"claim_text": "x" * 2001},
        {"claim_text": 1234567890},
        {"claim_text": QUERY, "limit": 0},
        {"claim_text": QUERY, "limit": 13},
        {"claim_text": QUERY, "limit": True},
        {"claim_text": QUERY, "limit": 1.5},
        {"claim_text": QUERY, "as_of": "not a datetime"},
        {"claim_text": QUERY, "as_of": "2024-01-01"},
        {"claim_text": QUERY, "as_of": 1704067200},
        {"claim_text": QUERY, "url": "https://arbitrary.example"},
    ],
)
async def test_check_validation_rejects_invalid_inputs_without_models(
    client, monkeypatch, body
):
    check = AsyncMock()
    monkeypatch.setattr(verdicts, "check_claim", check)
    response = await client.post("/api/v1/verdicts/check", json=body)
    assert response.status_code == 422
    check.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        EmbeddingUnavailableError(
            "Load compatible EMBEDDING_MODEL weights at startup."
        ),
        NLIUnavailableError("Pre-cache NLI_MODEL and its tokenizer dependencies."),
    ],
)
async def test_model_failures_return_actionable_503_not_fabricated_answer(
    client, monkeypatch, error
):
    monkeypatch.setattr(verdicts, "check_claim", AsyncMock(side_effect=error))
    response = await client.post("/api/v1/verdicts/check", json={"claim_text": QUERY})
    assert response.status_code == 503
    assert response.json()["detail"] == str(error)
    assert "assessment" not in response.json()


@pytest.mark.asyncio
async def test_database_failure_returns_actionable_503_without_raw_error(
    client, monkeypatch
):
    monkeypatch.setattr(
        verdicts,
        "check_claim",
        AsyncMock(side_effect=SQLAlchemyError("secret connection details")),
    )
    response = await client.post("/api/v1/verdicts/check", json={"claim_text": QUERY})
    assert response.status_code == 503
    assert "PostgreSQL/pgvector" in response.json()["detail"]
    assert "secret" not in response.text


@pytest.mark.asyncio
async def test_empty_page_preserves_total_in_one_snapshot(client, api_db):
    api_db.execute.return_value.mappings.return_value.all.return_value = [
        {"total": 7, "id": None}
    ]
    response = await client.get(
        "/api/v1/verdicts?limit=2&offset=100&band=UNRESOLVED&outlet=a.example"
    )
    assert response.status_code == 200
    assert response.json() == {
        "total": 7,
        "items": [],
        "limit": 2,
        "offset": 100,
        "has_more": False,
    }
    assert api_db.execute.await_count == 1
    statement, params = api_db.execute.call_args.args
    sql = str(statement)
    assert "COUNT(*) AS total FROM matching" in sql
    assert "c.verdict = :band" in sql and "a.domain = :outlet" in sql
    assert "ORDER BY verdict_probability ASC NULLS LAST, id ASC" in sql
    assert params == {
        "limit": 2,
        "offset": 100,
        "band": "UNRESOLVED",
        "outlet": "a.example",
    }
    api_db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_discloses_new_and_legacy_scores_without_rescoring(client, api_db):
    legacy = {"corroboration": {"value": 0.99}}
    api_db.execute.return_value.mappings.return_value.all.return_value = [
        {
            "total": 3,
            "id": str(UUID(int=1)),
            "probability": 0.9,
            "extracted_at": NOW,
            "score_evidence": legacy,
        },
        {
            "total": 3,
            "id": str(UUID(int=2)),
            "probability": 0.9,
            "extracted_at": NOW,
            "score_evidence": json.dumps(
                {
                    "method_version": METHOD_VERSION,
                    "score_kind": SCORE_KIND,
                    "warnings": ["Corpus-local only"],
                }
            ),
        },
    ]
    response = await client.get("/api/v1/verdicts?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3 and data["has_more"] is True
    assert data["items"][0]["probability"] == 0.9
    assert data["items"][0]["score_kind"] == "legacy_unverified"
    assert "Legacy" in data["items"][0]["warnings"][0]
    assert data["items"][1]["score_kind"] == SCORE_KIND
    assert data["items"][1]["method_version"] == METHOD_VERSION
    assert "score_evidence" not in data["items"][0]
    assert legacy == {"corroboration": {"value": 0.99}}
    api_db.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [None, "not JSON", {"old": "evidence"}])
async def test_detail_preserves_legacy_verdict_and_warns(client, api_db, stored):
    api_db.execute.return_value.mappings.return_value.first.return_value = {
        "id": str(UUID(int=1)),
        "verdict": "SUPPORTED",
        "probability": 0.95,
        "verdict_evidence": stored,
        "published_at": NOW,
    }
    response = await client.get(f"/api/v1/verdicts/{UUID(int=1)}")
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "SUPPORTED" and data["probability"] == 0.95
    assert data["score_kind"] == "legacy_unverified"
    assert any(
        "not been automatically rescored" in warning for warning in data["warnings"]
    )
    assert api_db.execute.await_count == 1
    api_db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_detail_current_version_discloses_uncalibrated_score(client, api_db):
    api_db.execute.return_value.mappings.return_value.first.return_value = {
        "id": str(UUID(int=1)),
        "verdict": "UNRESOLVED",
        "probability": 0.5,
        "verdict_evidence": {
            "method_version": METHOD_VERSION,
            "score_kind": SCORE_KIND,
        },
    }
    response = await client.get(f"/api/v1/verdicts/{UUID(int=1)}")
    assert response.status_code == 200
    assert response.json()["score_kind"] == SCORE_KIND
    assert "uncalibrated" in response.json()["score_disclosure"]


@pytest.mark.asyncio
async def test_malformed_detail_id_is_422_not_database_cast_error(client, api_db):
    response = await client.get("/api/v1/verdicts/not-a-uuid")
    assert response.status_code == 422
    api_db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_detail_is_404(client):
    response = await client.get(f"/api/v1/verdicts/{UUID(int=1)}")
    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/api/v1/verdicts", f"/api/v1/verdicts/{UUID(int=1)}"]
)
async def test_stored_verdict_storage_failure_is_503(client, api_db, path):
    api_db.execute.side_effect = SQLAlchemyError("private database error")
    response = await client.get(path)
    assert response.status_code == 503
    assert "private" not in response.text
