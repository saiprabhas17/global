"""Graph IDs, inferred edge metadata and bounded simulation contracts."""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.api.v1 import graph as graph_api
from app.db.neo4j_client import neo4j_client
from app.services.evolution.text_changes import analyze_text_change


def _id(n) -> str:
    return str(uuid.UUID(int=n))


def _result(rows) -> MagicMock:
    result = MagicMock()
    result.all.return_value = rows
    return result


def _database(monkeypatch, results) -> MagicMock:
    db = MagicMock()
    db.execute = AsyncMock(side_effect=results)
    context = AsyncMock()
    context.__aenter__.return_value = db
    monkeypatch.setattr("app.db.postgres.async_session_maker", lambda: context)
    return db


def _neo4j(monkeypatch, handler) -> MagicMock:
    async def run(query, **params):
        result = MagicMock()
        result.data = AsyncMock(return_value=handler(query, params))
        return result

    tx = MagicMock()
    tx.run = AsyncMock(side_effect=run)

    async def read(callback):
        return await callback(tx)

    session = MagicMock()
    session.execute_read = AsyncMock(side_effect=read)
    context = AsyncMock()
    context.__aenter__.return_value = session
    driver = MagicMock()
    driver.session.return_value = context
    monkeypatch.setattr(
        neo4j_client, "get_driver", AsyncMock(return_value=driver)
    )
    return tx


@pytest.mark.asyncio
async def test_claim_graph_keeps_claim_ids_and_exposes_article_ids_and_actual_edge_types(
    monkeypatch,
):
    analysis = analyze_text_change("10 people", "12 people")
    rows = [
        (_id(1), "10 people", "one.example", "UNSUPPORTED", 0.5, _id(101)),
        (_id(2), "12 people", "two.example", "UNSUPPORTED", 0.6, _id(102)),
    ]
    db = _database(
        monkeypatch,
        [_result(rows), _result([(_id(2), _id(1), 0.95, "EVOLVED_FROM", analysis)])],
    )
    result = await graph_api.claims_graph(limit=999999)
    assert [n["id"] for n in result["nodes"]] == [_id(1), _id(2)]
    assert [n["article_id"] for n in result["nodes"]] == [_id(101), _id(102)]
    edge = result["edges"][0]
    assert (edge["src"], edge["dst"]) == (_id(2), _id(1))
    assert edge["parent_claim_id"] == _id(1)
    assert edge["child_claim_id"] == _id(2)
    assert edge["relationship_type"] == "EVOLVED_FROM"
    assert "NUMERIC_DRIFT" in edge["analysis"]["mutation_types"]
    assert edge["observed_propagation"] is False
    assert result["edge_source"] == "claim_relationships"
    assert all(call.args[1]["lim"] == 3000 for call in db.execute.await_args_list)


@pytest.mark.asyncio
async def test_knn_fallback_is_similarity_not_invented_lineage(monkeypatch):
    rows = [
        (_id(1), "10 people", "one.example", "UNSUPPORTED", 0.5, _id(101)),
        (_id(2), "12 people", "two.example", "UNSUPPORTED", 0.6, _id(102)),
    ]
    _database(
        monkeypatch,
        [
            _result(rows),
            _result([]),
            _result([(_id(2), _id(1), 0.95, "SIMILAR_TO", None)]),
        ],
    )
    result = await graph_api.claims_graph()
    assert result["edge_source"] == "pgvector_knn"
    edge = result["edges"][0]
    assert edge["relationship_type"] == "SIMILAR_TO"
    assert edge["parent_claim_id"] is edge["child_claim_id"] is None
    assert edge["analysis"] is None
    assert edge["observed_propagation"] is False


@pytest.mark.parametrize("selected", [_id(1), _id(101)])
@pytest.mark.asyncio
async def test_timeline_resolves_legacy_claim_id_to_article_id_without_relabeling_graph_nodes(
    monkeypatch, selected
):
    result = MagicMock()
    result.scalar_one_or_none.return_value = _id(101)
    _database(monkeypatch, [result])
    tx = _neo4j(monkeypatch, lambda query, params: [])
    response = await graph_api.timeline_clusters(article_id=selected, limit=999999)
    assert response["focused"] is True
    assert response["article_id"] == _id(101)
    params = tx.run.await_args.kwargs
    assert params["aid"] == _id(101)
    assert params["lim"] == 120
    assert "SIMILAR*1..3" in tx.run.await_args.args[0]


@pytest.mark.asyncio
async def test_timeline_rejects_invalid_id_before_any_graph_access(monkeypatch):
    get_driver = AsyncMock()
    monkeypatch.setattr(neo4j_client, "get_driver", get_driver)
    with pytest.raises(HTTPException) as error:
        await graph_api.timeline_clusters(article_id="not-an-id")
    assert error.value.status_code == 400
    get_driver.assert_not_awaited()


def _simulation_handler(links, *, exists=True) -> Any:
    def handler(query, params):
        if "RETURN a.title" in query:
            return [{"title": "Hub story", "domain": "one.example"}] if exists else []
        return [
            {"src": a, "dst": b} for a, b in sorted(links) if a in params["frontier"]
        ]

    return handler


@pytest.mark.asyncio
async def test_simulation_only_reads_bounded_hub_frontiers_and_deduplicates_undirected_edges(
    monkeypatch,
):
    links = [
        ("a", "b"),
        ("b", "a"),
        ("b", "c"),
        ("c", "b"),
        ("c", "d"),
        ("d", "c"),
        ("unrelated", "other"),
    ]
    tx = _neo4j(monkeypatch, _simulation_handler(links))
    response = await graph_api.simulate_spread("a", max_depth=999999)
    assert response["max_depth"] == graph_api.MAX_SIMULATION_DEPTH
    assert response["deterministic_reach"] == 4
    assert response["mean_fanout"] == 1.5  # Three undirected edges, no double counting.
    assert response["graph_scope"] == "bounded_hub_neighborhood"
    assert response["observed_propagation"] is False
    for call in tx.run.await_args_list[1:]:
        assert "$frontier" in call.args[0]
        assert "LIMIT $lim" in call.args[0]
        assert len(call.kwargs["frontier"]) <= graph_api.MAX_SIMULATION_NODES
        assert call.kwargs["lim"] == graph_api.MAX_SIMULATION_EDGES + 1
        assert "unrelated" not in call.kwargs["frontier"]


@pytest.mark.asyncio
async def test_simulation_depth_minimum_and_probability_clamp(monkeypatch):
    tx = _neo4j(monkeypatch, _simulation_handler([("a", "b"), ("b", "c")]))
    response = await graph_api.simulate_spread("a", p=2.0, max_depth=-100)
    assert response["max_depth"] == 1
    assert response["p"] == 0.95
    assert response["deterministic_reach"] == 2
    assert tx.run.await_count == 2  # Hub lookup + one bounded frontier.


@pytest.mark.asyncio
async def test_simulation_enforces_node_and_edge_caps_and_reports_partial_result(
    monkeypatch,
):
    monkeypatch.setattr(graph_api, "MAX_SIMULATION_NODES", 3)
    monkeypatch.setattr(graph_api, "MAX_SIMULATION_EDGES", 2)
    _neo4j(
        monkeypatch,
        _simulation_handler([("a", "b"), ("a", "c"), ("a", "d"), ("a", "e")]),
    )
    response = await graph_api.simulate_spread("a")
    assert response["deterministic_reach"] == 3
    assert response["truncated"] is True
    assert response["limits"]["nodes"] == 3
    assert response["limits"]["edges"] == 2


@pytest.mark.asyncio
async def test_simulation_missing_hub_does_not_scan_graph(monkeypatch):
    tx = _neo4j(monkeypatch, _simulation_handler([], exists=False))
    with pytest.raises(HTTPException) as error:
        await graph_api.simulate_spread("missing")
    assert error.value.status_code == 404
    assert tx.run.await_count == 1
