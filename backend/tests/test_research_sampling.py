"""Deterministic real-list sampling and idempotent, non-destructive DB writes."""

import random
import uuid
from collections import Counter
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.dialects import postgresql

from scripts.sample_eval_pairs import fetch_pool, main, pick_pairs, pool_fingerprint


def pool(size: int = 30, identical: bool = False) -> list[tuple[str, NDArray[np.float64], str]]:
    rng = np.random.default_rng(42)
    return [(str(uuid.UUID(int=100 + i)), np.asarray([1.0, 0.0, 0.0]) if identical else rng.normal(size=3), f"outlet-{i % 3}") for i in range(size)]


def test_pairs_deterministic_canonical_and_global_claim_cap():
    corpus = pool()
    selected = pick_pairs(corpus, 50, random.Random(20260916))
    assert selected == pick_pairs(list(reversed(corpus)), 50, random.Random(20260916))
    usage = Counter(claim for a, b, _ in selected for claim in (a, b))
    assert usage and max(usage.values()) <= 2
    assert all(a < b for a, b, _ in selected)
    assert len({(a, b) for a, b, _ in selected}) == len(selected)


def test_rng_shuffles_actual_candidates_and_changes_seeded_selection():
    corpus = pool(size=12, identical=True)
    a = pick_pairs(corpus, 4, random.Random(1))
    b = pick_pairs(corpus, 4, random.Random(2))
    assert a != b
    domains = {claim: domain for claim, _, domain in corpus}
    assert all(domains[x] != domains[y] for x, y, _ in a)


def test_pool_hash_order_independent_and_tracks_embeddings():
    corpus = pool()
    assert pool_fingerprint(corpus) == pool_fingerprint(list(reversed(corpus)))
    changed = list(corpus)
    changed[0] = (changed[0][0], changed[0][1] + 0.01, changed[0][2])
    assert pool_fingerprint(corpus) != pool_fingerprint(changed)


def test_empty_pool_and_zero_vectors_do_not_create_fake_similarity():
    assert pick_pairs([], 2, random.Random(1)) == []
    corpus = [(str(uuid.UUID(int=i + 1)), np.zeros(3), "outlet") for i in range(3)]
    assert pick_pairs(corpus, 2, random.Random(1)) == []


@pytest.mark.asyncio
async def test_database_pool_is_seeded_stable_not_sql_random(monkeypatch):
    db = AsyncMock()
    db.execute.return_value = SimpleNamespace(all=list)
    context = AsyncMock()
    context.__aenter__.return_value = db
    monkeypatch.setattr("scripts.sample_eval_pairs.async_session_maker", lambda: context)
    assert await fetch_pool(100, 12345) == []
    statement, params = db.execute.call_args.args
    sql = str(statement)
    assert "ORDER BY md5" in sql and ", c.id" in sql
    assert "random()" not in sql
    assert params == {"n": 100, "seed": "12345"}


class FakeResult:
    def __init__(self, rows: list[Any] | None = None, value: Any = None) -> None:
        self.rows = rows or []
        self.value = value

    def all(self) -> list[Any]:
        return self.rows

    def scalar_one_or_none(self) -> Any:
        return self.value


class PairStore:
    def __init__(self) -> None:
        self.pairs: dict[str, dict[str, Any]] = {}
        self.statements: list[str] = []
        self.commits = 0

    def get_bind(self) -> SimpleNamespace:
        return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

    async def execute(self, statement: Any) -> FakeResult:
        self.statements.append(str(statement))
        if getattr(statement, "is_insert", False):
            values = statement.compile(dialect=cast(Any, postgresql.dialect)()).params
            key = values["canonical_key"]
            if key in self.pairs:
                return FakeResult()
            self.pairs[key] = values
            return FakeResult(value=values["id"])
        if "SELECT claim_a_id::text, claim_b_id::text, split FROM eval_pairs" in str(statement):
            return FakeResult(rows=[(str(p["claim_a_id"]), str(p["claim_b_id"]), p["split"]) for p in self.pairs.values()])
        return FakeResult()

    async def commit(self) -> None:
        self.commits += 1


@pytest.mark.asyncio
async def test_sampler_rerun_is_idempotent_and_never_deletes_votes(monkeypatch):
    store = PairStore()
    context = AsyncMock()
    context.__aenter__.return_value = store
    corpus = pool(size=16)
    monkeypatch.setattr("scripts.sample_eval_pairs.async_session_maker", lambda: context)
    monkeypatch.setattr("scripts.sample_eval_pairs.fetch_pool", AsyncMock(return_value=corpus))
    monkeypatch.setattr("sys.argv", ["sample_eval_pairs.py", "5", "16", "--seed", "123"])
    await main()
    first = dict(store.pairs)
    assert first
    await main()
    assert first == store.pairs and store.commits == 2
    usage = Counter(str(p[key]) for p in store.pairs.values() for key in ("claim_a_id", "claim_b_id"))
    assert max(usage.values()) <= 2
    assert all(p["sampling_provenance"]["seed"] == 123 for p in store.pairs.values())
    assert all(p["split"] == p["event_group"] == "unassigned" for p in store.pairs.values())
    assert not any("DELETE" in statement.upper() for statement in store.statements)
