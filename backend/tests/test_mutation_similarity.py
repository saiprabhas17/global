"""Bounded memory candidate searches and link-graph input caps."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.services.evolution import similarity
from scripts import build_link_graph


def test_blockwise_neighbors_match_dense_reference_on_small_input():
    vectors = np.random.default_rng(42).normal(size=(21, 7)).astype(np.float32)
    unit = similarity.unit_matrix(vectors)
    dense = unit @ unit.T  # Small test-only oracle; production must not do this.
    for i in range(len(unit)):
        expected = sorted(
            [(j, float(dense[i, j])) for j in range(i) if dense[i, j] >= -0.1],
            key=lambda pair: (-pair[1], pair[0]),
        )[:4]
        actual = similarity.top_neighbors(
            unit, i, limit=4, threshold=-0.1, stop=i, block_size=3
        )
        assert [j for j, _ in actual] == [j for j, _ in expected]
        assert [score for _, score in actual] == pytest.approx(
            [score for _, score in expected], abs=1e-6
        )


def test_candidate_search_never_materializes_square_similarity(monkeypatch):
    unit = similarity.unit_matrix(np.ones((257, 4), dtype=np.float32))
    original = np.matmul
    shapes = []

    def bounded_matmul(left, right):
        shapes.append((left.shape, right.shape))
        assert left.shape[0] <= 11
        assert right.ndim == 1
        result = original(left, right)
        assert result.ndim == 1
        return result

    monkeypatch.setattr(np, "matmul", bounded_matmul)
    neighbors = similarity.top_neighbors(
        unit, 256, limit=5, threshold=0.8, block_size=11
    )
    assert len(neighbors) == 5
    assert [j for j, _ in neighbors] == [0, 1, 2, 3, 4]
    assert shapes


def test_ties_and_self_exclusion_are_deterministic():
    unit = similarity.unit_matrix([[1, 0]] * 5)
    assert [
        j
        for j, _ in similarity.top_neighbors(
            unit, 0, limit=2, threshold=0.8, block_size=1
        )
    ] == [1, 2]
    assert [
        j for j, _ in similarity.top_neighbors(unit, 4, limit=2, threshold=0.8, start=2)
    ] == [2, 3]


@pytest.mark.parametrize("count", [0, 1])
def test_link_graph_handles_empty_and_singleton_inputs(count):
    edges, duplicates = build_link_graph.find_edges(
        np.zeros((count, 3), dtype=np.float32), [""] * count
    )
    assert edges == duplicates == []


def test_link_graph_edges_match_dense_topk_and_deduplicate_pairs():
    unit = similarity.unit_matrix([[1, 0], [1, 0], [0.8, 0.6], [0, 1]])
    dense = unit @ unit.T
    expected = {}
    for i in range(len(unit)):
        candidates = sorted(
            [
                (j, float(dense[i, j]))
                for j in range(len(unit))
                if i != j and dense[i, j] >= 0.75
            ],
            key=lambda pair: (-pair[1], pair[0]),
        )[:2]
        for j, score in candidates:
            expected[(min(i, j), max(i, j))] = round(score, 4)
    edges, duplicates = build_link_graph.find_edges(
        unit, ["same", "same", "same", ""], top_k=2, threshold=0.75, block_size=1
    )
    assert {(i, j): score for i, j, score in edges} == expected
    assert len({(i, j) for i, j, _ in edges}) == len(edges)
    assert duplicates == [(0, 1), (0, 2)]  # Linear star, not a dense duplicate clique.


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"block_size": 0}])
def test_link_graph_rejects_unbounded_or_invalid_candidate_settings(kwargs):
    with pytest.raises(ValueError):
        build_link_graph.find_edges(np.ones((2, 3)), ["", ""], **kwargs)


def test_link_graph_cli_and_environment_expose_positive_caps(monkeypatch):
    monkeypatch.setenv("LINK_GRAPH_MAX_ARTICLES", "17")
    monkeypatch.setenv("LINK_GRAPH_BLOCK_SIZE", "8")
    args = build_link_graph.parse_args([])
    assert (args.max_articles, args.block_size) == (17, 8)
    args = build_link_graph.parse_args(["--max-articles", "23", "--block-size", "3"])
    assert (args.max_articles, args.block_size) == (23, 3)
    with pytest.raises(SystemExit):
        build_link_graph.parse_args(["--max-articles", "0"])


@pytest.mark.asyncio
async def test_link_graph_load_is_bounded_in_sql_and_reports_truncation(
    monkeypatch, capsys
):
    rows = [
        SimpleNamespace(_mapping={"id": str(i), "title": "title"}) for i in range(4)
    ]
    result = MagicMock()
    result.all.return_value = rows
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    context = AsyncMock()
    context.__aenter__.return_value = db
    monkeypatch.setattr(build_link_graph, "async_session_maker", lambda: context)
    articles = await build_link_graph.load_articles(max_articles=3)
    assert len(articles) == 3
    query, params = db.execute.await_args.args
    assert "LIMIT :max_articles" in str(query)
    assert "2000" in str(query)
    assert params["max_articles"] == 4  # One sentinel row detects truncation.
    assert "cap reached" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_empty_link_graph_build_does_not_load_a_model_or_write_graph(monkeypatch):
    monkeypatch.setattr(build_link_graph, "load_articles", AsyncMock(return_value=[]))
    embed, write = MagicMock(), MagicMock()
    monkeypatch.setattr(build_link_graph, "embed_all", embed)
    monkeypatch.setattr(build_link_graph, "write_neo4j", write)
    await build_link_graph.main(max_articles=10)
    embed.assert_not_called()
    write.assert_not_called()
