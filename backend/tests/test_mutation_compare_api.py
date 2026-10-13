"""Authenticated bounded comparison API, with no database/model inference."""

import uuid
from unittest.mock import MagicMock

import pytest

from app.api.deps import get_current_user
from app.api.v1 import graph as graph_api
from app.main import app
from app.models.user import RoleEnum, User


def _authenticate(monkeypatch) -> None:
    monkeypatch.setitem(
        app.dependency_overrides,
        get_current_user,
        lambda: User(
            id=uuid.uuid4(),
            email="reader@example.com",
            role=RoleEnum.user,
        ),
    )


@pytest.mark.asyncio
async def test_compare_requires_current_user_before_analysis(async_client, monkeypatch):
    analyzer = MagicMock()
    monkeypatch.setattr(graph_api, "analyze_text_change", analyzer)
    response = await async_client.post(
        "/api/v1/graph/mutation/compare",
        json={"older_text": "10 people", "newer_text": "12 people"},
    )
    assert response.status_code == 401
    analyzer.assert_not_called()


@pytest.mark.asyncio
async def test_compare_returns_typed_analysis_and_temporal_metadata(
    async_client, monkeypatch
):
    _authenticate(monkeypatch)
    body = {
        "older_text": "The agency reportedly found 10 survivors.",
        "newer_text": "The agency found 12 survivors.",
        "older_timestamp": "2026-09-20T12:00:00Z",
        "newer_timestamp": "2026-09-20T13:00:00Z",
    }
    response = await async_client.post("/api/v1/graph/mutation/compare", json=body)
    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert {"NUMERIC_DRIFT", "HEDGING_SHIFT"} <= set(analysis["mutation_types"])
    assert analysis["observed_propagation"] is False
    assert analysis["algorithm_version"]
    assert analysis["temporal_order"] == "strictly_older"
    assert analysis["lag_seconds"] == 3600
    assert analysis["limitations"]
    for span in analysis["changed_spans"]:
        for label in ("older", "newer"):
            source, change = body[f"{label}_text"], span[f"{label}_span"]
            assert source[change["start"] : change["end"]] == change["text"]


@pytest.mark.parametrize(
    "body",
    [
        {"older_text": "x" * 2001, "newer_text": "ok"},
        {"older_text": "ok", "newer_text": "x" * 2001},
        {"older_text": " ", "newer_text": "ok"},
        {"older_text": "ok", "newer_text": ""},
        {"older_text": "ok"},
        {"older_text": "ok", "newer_text": "ok", "older_timestamp": "not-a-date"},
    ],
)
@pytest.mark.asyncio
async def test_compare_rejects_invalid_inputs(async_client, monkeypatch, body):
    _authenticate(monkeypatch)
    response = await async_client.post("/api/v1/graph/mutation/compare", json=body)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_compare_accepts_exact_limit_and_reports_unknown_time(
    async_client, monkeypatch
):
    _authenticate(monkeypatch)
    text = "x" * 2000
    response = await async_client.post(
        "/api/v1/graph/mutation/compare", json={"older_text": text, "newer_text": text}
    )
    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["mutation_types"] == ["NEAR_DUPLICATE"]
    assert analysis["meaningful_change"] is False
    assert analysis["temporal_order"] == "unknown"


@pytest.mark.asyncio
async def test_compare_does_not_invent_timestamp_order(async_client, monkeypatch):
    _authenticate(monkeypatch)
    response = await async_client.post(
        "/api/v1/graph/mutation/compare",
        json={
            "older_text": "10 people",
            "newer_text": "12 people",
            "older_timestamp": "2026-09-21T00:00:00Z",
            "newer_timestamp": "2026-09-20T00:00:00Z",
        },
    )
    assert response.status_code == 200
    assert response.json()["analysis"]["temporal_order"] == "reversed"
    assert response.json()["analysis"]["observed_propagation"] is False
