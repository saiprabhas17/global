"""Lineage contracts: actual persisted edges, never time-adjacent siblings."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.api.v1 import lineage as lineage_api
from app.services.evolution.text_changes import analyze_text_change

NOW = datetime(2026, 9, 20, tzinfo=UTC)


def _id(n) -> str:
    return str(uuid.UUID(int=n))


def _version(n, day, text) -> tuple[Any, ...]:
    return (
        _id(n),
        text,
        "example.com",
        NOW + timedelta(days=day),
        f"Article {n}",
        _id(1000 + n),
    )


def _edge(child, parent, evidence=None, score=0.95) -> tuple[Any, ...]:
    return (_id(child), _id(parent), score, evidence, _id(5000 + child), NOW)


def _mock_lineage_db(monkeypatch, edge_rows, version_rows) -> MagicMock:
    async def execute(statement, params):
        ids = {value for key, value in params.items() if key.startswith("i")}
        result = MagicMock()
        if "FROM claim_relationships" in str(statement):
            rows = [row for row in edge_rows if row[0] in ids or row[1] in ids]
            rows.sort(key=lambda row: (-row[2], row[0], row[1]))
            result.all.return_value = rows[: params["edge_lim"]]
        else:
            rows = [row for row in version_rows if row[0] in ids]
            rows.sort(
                key=lambda row: (
                    row[3] is None,
                    row[3] or datetime.max.replace(tzinfo=UTC),
                    row[0],
                )
            )
            result.all.return_value = rows
        return result

    db = MagicMock()
    db.execute = AsyncMock(side_effect=execute)
    context = AsyncMock()
    context.__aenter__.return_value = db
    monkeypatch.setattr(lineage_api, "async_session_maker", lambda: context)
    return db


def test_connected_version_cap_does_not_select_detached_high_score_siblings(
    monkeypatch,
):
    monkeypatch.setattr(lineage_api, "MAX_VERSIONS", 3)
    edges = [
        {"from": "a", "to": "root", "score": 0.3},
        {"from": "b", "to": "a", "score": 0.99},
        {"from": "c", "to": "b", "score": 0.98},
    ]
    assert lineage_api._connected_versions("root", edges) == ["root", "a", "b"]


def test_clean_strips_html_and_collapses_whitespace():
    assert lineage_api._clean("<p>Hello <b>world</b></p>", 100) == "Hello world"
    assert lineage_api._clean("  a\n\n   b  ", 100) == "a b"
    assert lineage_api._clean("Tom &amp; Jerry", 100) == "Tom & Jerry"


def test_clean_drops_empty_and_truncates():
    assert lineage_api._clean(None, 100) is None
    assert lineage_api._clean("   ", 100) is None
    assert lineage_api._clean("abcdef", 3) == "abc"


def test_chain_caps_are_ordered():
    assert 1 < lineage_api.MAX_VERSIONS <= lineage_api.MAX_COMPONENT
    assert lineage_api.MAX_HOPS >= 1
    assert lineage_api.MAX_EDGE_ROWS >= lineage_api.MAX_COMPONENT


@pytest.mark.asyncio
async def test_lineage_rejects_non_uuid(async_client):
    resp = await async_client.get("/api/v1/graph/lineage/not-a-uuid")
    assert resp.status_code == 400


def test_lineage_compare_and_simulate_routes_are_mounted():
    from app.main import app

    paths = set(app.openapi()["paths"])
    assert "/api/v1/graph/lineage/{claim_id}" in paths
    assert "/api/v1/graph/simulate" in paths
    assert "/api/v1/graph/mutation/compare" in paths


@pytest.mark.asyncio
async def test_branching_lineage_diffs_follow_actual_parents_not_sorted_siblings(
    monkeypatch,
):
    versions = [
        _version(1, 0, "The crew reportedly found 10 people."),
        _version(2, 1, "The crew found 12 people."),
        _version(3, 2, "The crew found 14 people."),
    ]
    stored = analyze_text_change(
        versions[0][1],
        versions[1][1],
        older_timestamp=versions[0][3],
        newer_timestamp=versions[1][3],
    )
    db = _mock_lineage_db(monkeypatch, [_edge(2, 1, stored), _edge(3, 1)], versions)
    result = await lineage_api.claim_lineage(_id(1))
    assert {"root", "versions", "diffs", "edges", "counts"} <= result.keys()
    assert [(v["id"], v["article_id"]) for v in result["versions"]] == [
        (_id(i), _id(1000 + i)) for i in (1, 2, 3)
    ]
    assert {(d["from_id"], d["to_id"]) for d in result["diffs"]} == {
        (_id(1), _id(2)),
        (_id(1), _id(3)),
    }
    assert {(d["from_index"], d["to_index"]) for d in result["diffs"]} == {
        (0, 1),
        (0, 2),
    }
    assert (_id(2), _id(3)) not in {(d["from_id"], d["to_id"]) for d in result["diffs"]}
    assert {(e["parent_claim_id"], e["child_claim_id"]) for e in result["edges"]} == {
        (_id(1), _id(2)),
        (_id(1), _id(3)),
    }
    assert {e["evidence_source"] for e in result["edges"]} == {
        "persisted",
        "legacy_edge_reanalysis",
    }
    for edge in result["edges"]:
        assert edge["from"] == edge["child_claim_id"]
        assert edge["to"] == edge["parent_claim_id"]
        assert edge["analysis"]["observed_propagation"] is False
        assert "HEDGING_SHIFT" in edge["analysis"]["mutation_types"]
    assert result["counts"]["edges"] == 2
    assert all(
        "LIMIT :edge_lim" in str(call.args[0])
        for call in db.execute.await_args_list
        if "FROM claim_relationships" in str(call.args[0])
    )


@pytest.mark.asyncio
async def test_lineage_preserves_stored_algorithm_evidence(monkeypatch):
    versions = [_version(1, 0, "10 people"), _version(2, 1, "12 people")]
    stored = analyze_text_change(versions[0][1], versions[1][1])
    stored["algorithm_version"] = "historical-version"
    _mock_lineage_db(monkeypatch, [_edge(2, 1, stored)], versions)
    result = await lineage_api.claim_lineage(_id(2))
    assert result["edges"][0]["analysis"]["algorithm_version"] == "historical-version"
    assert result["diffs"][0]["analysis"]["algorithm_version"] == "historical-version"
    assert result["edges"][0]["evidence_source"] == "persisted"


@pytest.mark.asyncio
async def test_legacy_equal_time_edge_is_exposed_but_not_claimed_observed(monkeypatch):
    versions = [_version(1, 0, "10 people"), _version(2, 0, "12 people")]
    _mock_lineage_db(monkeypatch, [_edge(2, 1)], versions)
    result = await lineage_api.claim_lineage(_id(1))
    assert result["edges"][0]["analysis"]["temporal_order"] == "equal"
    assert result["edges"][0]["observed_propagation"] is False
    assert result["edges"][0]["evidence_source"] == "legacy_edge_reanalysis"


@pytest.mark.asyncio
async def test_lineage_component_limit_is_hard_and_edges_reference_rendered_versions(
    monkeypatch,
):
    monkeypatch.setattr(lineage_api, "MAX_COMPONENT", 4)
    monkeypatch.setattr(lineage_api, "MAX_VERSIONS", 4)
    versions = [_version(i, i, f"{10 + i} people") for i in range(1, 12)]
    _mock_lineage_db(monkeypatch, [_edge(i, 1) for i in range(2, 12)], versions)
    result = await lineage_api.claim_lineage(_id(1))
    assert result["counts"]["component_claims"] == 4
    assert result["counts"]["versions"] == 4
    assert result["truncated"] is True
    visible = {v["id"] for v in result["versions"]}
    assert all(
        {e["parent_claim_id"], e["child_claim_id"]} <= visible for e in result["edges"]
    )
    assert len(result["diffs"]) == len(result["edges"]) == 3


@pytest.mark.asyncio
async def test_no_actual_lineage_is_404(monkeypatch):
    _mock_lineage_db(monkeypatch, [], [_version(1, 0, "10 people")])
    with pytest.raises(HTTPException) as error:
        await lineage_api.claim_lineage(_id(1))
    assert error.value.status_code == 404
