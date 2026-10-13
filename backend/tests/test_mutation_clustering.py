"""Clustering filtering, CPU isolation and version metadata without real models."""

import asyncio
import sys
import threading
import uuid
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.models.claim import Claim
from app.models.evolution import ClaimClusterAssignment, ClaimClusterRun
from app.services.evolution import cluster_service
from app.services.evolution.orchestrator import EvolutionOrchestrator


def _db() -> MagicMock:
    db = MagicMock()

    async def flush():
        for call in db.add.call_args_list:
            obj = call.args[0]
            if isinstance(obj, ClaimClusterRun) and obj.id is None:
                obj.id = uuid.UUID(int=100)

    db.flush = AsyncMock(side_effect=flush)
    return db


def _claims() -> list[Claim]:
    return [
        Claim(id=uuid.UUID(int=1), claim_text="first", embedding=[1.0, 0.0]),
        Claim(id=uuid.UUID(int=2), claim_text="missing", embedding=None),
        Claim(id=uuid.UUID(int=3), claim_text="last", embedding=[0.0, 1.0]),
    ]


@pytest.mark.asyncio
async def test_clustering_excludes_null_embeddings_and_fitting_does_not_block_loop(
    monkeypatch,
):
    service, db = cluster_service.ClusterService(), _db()
    started, release = threading.Event(), threading.Event()
    captured: dict[str, Any] = {}
    main_thread = threading.get_ident()

    def fit(texts, embeddings, params):
        captured.update(
            texts=texts,
            embeddings=embeddings,
            params=params,
            thread=threading.get_ident(),
        )
        started.set()
        assert release.wait(timeout=2), "clustering blocked the event loop"
        return cluster_service._ClusterResult([0, -1], {0: "test topic"}, {0: ["test"]})

    monkeypatch.setattr(cluster_service, "_fit_topics", fit)
    task = asyncio.create_task(service.run_clustering(db, _claims()))
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        assert captured["thread"] != main_thread
        release.set()
        run = await asyncio.wait_for(task, timeout=3)
    finally:
        release.set()
        if not task.done():
            await task
    assert captured["texts"] == ["first", "last"]
    assert captured["embeddings"] == [[1.0, 0.0], [0.0, 1.0]]
    assert run.claims_in_corpus == 2
    assert (
        run.algorithm_params["algorithm_version"]
        == cluster_service.CLUSTERING_ALGORITHM_VERSION
    )
    assert run.algorithm_params["excluded_missing_embeddings"] == 1
    assert run.algorithm_params["corpus_count_basis"] == "embedded_claims"
    assignments = db.add_all.call_args.args[0]
    assert [(a.claim_id, a.topic_id) for a in assignments] == [
        (uuid.UUID(int=1), 0),
        (uuid.UUID(int=3), -1),
    ]
    assert all(a.cluster_run_id == run.id for a in assignments)


@pytest.mark.asyncio
async def test_empty_embedded_corpus_creates_no_assignments_or_model_import(
    monkeypatch,
):
    service, db = cluster_service.ClusterService(), _db()
    monkeypatch.setitem(sys.modules, "bertopic", None)
    monkeypatch.setitem(sys.modules, "umap", None)
    run = await service.run_clustering(
        db, [Claim(id=uuid.UUID(int=1), claim_text="missing", embedding=None)]
    )
    assert run.claims_in_corpus == 0
    assert db.add_all.call_args.args[0] == []
    assert run.algorithm_params["excluded_missing_embeddings"] == 1


def test_fit_uses_versioned_parameters_and_supplied_embeddings_only(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeTopicModel:
        def __init__(self, **kwargs):
            captured["bertopic"] = kwargs

        def fit_transform(self, *, documents, embeddings):
            captured.update(documents=documents, embeddings=embeddings)
            return [0, 0, -1, 0, 0, 0], None

        def get_topic_info(self):
            return SimpleNamespace(
                iterrows=lambda: iter(
                    [
                        (0, {"Topic": 0, "Name": "topic", "Representation": ["word"]}),
                        (1, {"Topic": -1, "Name": "noise", "Representation": []}),
                    ]
                )
            )

    bertopic = ModuleType("bertopic")
    monkeypatch.setattr(bertopic, "BERTopic", FakeTopicModel, raising=False)
    umap = ModuleType("umap")
    fake_umap = MagicMock(return_value="fake_umap")
    monkeypatch.setattr(umap, "UMAP", fake_umap, raising=False)
    monkeypatch.setitem(sys.modules, "bertopic", bertopic)
    monkeypatch.setitem(sys.modules, "umap", umap)
    params = {
        "min_topic_size": 5,
        "umap_n_neighbors": 5,
        "umap_n_components": 4,
        "umap_min_dist": 0.0,
        "umap_metric": "cosine",
        "random_state": 42,
    }
    result = cluster_service._fit_topics(["doc"] * 6, [[1.0, 0.0]] * 6, params)
    assert result.topics == [0, 0, -1, 0, 0, 0]
    assert result.labels == {0: "topic"}
    assert result.keywords == {0: ["word"]}
    assert captured["embeddings"].dtype == np.float32
    assert captured["bertopic"]["calculate_probabilities"] is False
    assert captured["bertopic"]["min_topic_size"] == 5
    assert fake_umap.call_args.kwargs["random_state"] == 42


@pytest.mark.asyncio
async def test_orchestrator_counts_embedded_claims_and_syncs_noise_and_missing_children(
    monkeypatch,
):
    orchestrator = EvolutionOrchestrator()
    claims = [
        Claim(
            id=uuid.UUID(int=1),
            article_id=uuid.UUID(int=101),
            claim_text="noise",
            embedding=[1, 0],
        ),
        Claim(
            id=uuid.UUID(int=2),
            article_id=uuid.UUID(int=102),
            claim_text="missing",
            embedding=None,
        ),
    ]
    assignment = ClaimClusterAssignment(claim_id=claims[0].id, topic_id=-1)
    results = []
    for rows in (claims, [], [], [assignment]):
        result = MagicMock()
        result.scalars.return_value.all.return_value = rows
        results.append(result)
    db = MagicMock()
    db.scalar = AsyncMock(return_value=40)
    db.execute = AsyncMock(side_effect=results)
    db.commit, db.refresh = AsyncMock(), AsyncMock()
    cluster_mock = AsyncMock(return_value=ClaimClusterRun(id=uuid.UUID(int=200), claims_in_corpus=1))
    monkeypatch.setattr(orchestrator.cluster_service, "run_clustering", cluster_mock)
    monkeypatch.setattr(orchestrator.mutation_detector, "run_mutation_detection", AsyncMock(return_value=[]))
    graph_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(orchestrator.neo4j_writer, "sync_to_graph", graph_mock)
    outcome = await orchestrator.run_evolution_cycle(db, force=True)
    assert outcome["status"] == "success"
    run = cluster_mock.return_value
    assert run.algorithm_params["mutation_algorithm_version"]
    assert run.algorithm_params["text_change_algorithm_version"]
    assert run.algorithm_params["mutation_max_lag_days"] == 30
    assert run.algorithm_params["mutation_candidate_limit"] == 50
    assert "embedding IS NOT NULL" in str(db.scalar.await_args.args[0])
    assert graph_mock.call_args.kwargs[
        "evaluated_claim_ids"
    ] == {c.id for c in claims}
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_historical_algorithm_does_not_debounce_typed_upgrade_indefinitely(monkeypatch):
    orchestrator = EvolutionOrchestrator()
    db = MagicMock()
    db.scalar = AsyncMock(
        side_effect=[
            40,
            ClaimClusterRun(
                claims_in_corpus=50,
                algorithm_params={"min_topic_size": 5},
            ),
        ]
    )
    empty = MagicMock()
    empty.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=empty)
    db.commit, db.refresh = AsyncMock(), AsyncMock()
    cluster_mock = AsyncMock(return_value=ClaimClusterRun(id=uuid.UUID(int=200), claims_in_corpus=40))
    monkeypatch.setattr(orchestrator.cluster_service, "run_clustering", cluster_mock)
    monkeypatch.setattr(orchestrator.mutation_detector, "run_mutation_detection", AsyncMock(return_value=[]))
    graph_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(orchestrator.neo4j_writer, "sync_to_graph", graph_mock)
    result = await orchestrator.run_evolution_cycle(db)
    assert result["status"] == "success"
    cluster_mock.assert_awaited_once()
    # The old all-claims corpus count exceeds the embedded count, but an
    # algorithm upgrade still reconciles old mutation semantics once.
    assert db.scalar.await_count == 2
