"""The annotator-blinding contract for the gold-standard evaluation API.

`GET /eval/next` must never leak the stored similarity score, bucket or any
engine verdict: an annotator who sees them labels the machine's opinion, not
the claims. These tests pin that invariant (plus the DB-free validation paths
and route registration) so the blind cannot be removed by accident.

Authenticated label/selection/progress/split/export paths are exercised on
SQLite; PostgreSQL-specific snapshot/locking behavior is separately pinned
without connecting to a live database.
"""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import select

from app.api.deps import get_current_user
from app.api.v1.eval import VALID_LABELS, EvalLabelBody, _blinded, _per_annotator
from app.main import app
from app.models.eval import EvalPairLabel
from app.models.user import RoleEnum
from app.services.eval.dataset import iso
from tests.test_research_support import seed_pair, seed_user

PAIR_ID = "11111111-1111-1111-1111-111111111111"


def _fake_row(**extra: object) -> SimpleNamespace:
    """A row shaped like the one _PAIR_SQL returns (plus optional cues)."""
    row = SimpleNamespace(
        id=PAIR_ID,
        text_a="claim A",
        domain_a="a.example",
        text_b="claim B",
        domain_b="b.example",
    )
    for key, value in extra.items():
        setattr(row, key, value)
    return row


def test_blinded_returns_only_texts_and_domains():
    payload = _blinded(_fake_row(bucket="b85", sim_score=0.93, verdict="DISPUTED"))
    assert set(payload) == {"pair_id", "a", "b"}
    assert payload["pair_id"] == PAIR_ID
    assert payload["a"] == {"text": "claim A", "domain": "a.example"}
    assert payload["b"] == {"text": "claim B", "domain": "b.example"}


def test_blinded_payload_carries_no_engine_cues():
    payload = _blinded(
        _fake_row(bucket="b85", sim_score=0.93, verdict="DISPUTED", stance="CONTRADICT")
    )
    flat = repr(payload).lower()
    for cue in ("0.93", "b85", "score", "bucket", "verdict", "stance"):
        assert cue not in flat


def test_every_declared_label_is_accepted():
    for label in VALID_LABELS:
        body = EvalLabelBody.model_validate({"pair_id": PAIR_ID, "annotator": "tester", "label": label})
        assert body.label == label
        assert body.pair_id == uuid.UUID(PAIR_ID)


def test_per_annotator_keeps_every_label_per_annotator():
    """Regression: flattening to one {label, count} lost all but the last vote."""
    out = _per_annotator(
        [
            ("tester1", "DISTINCT", 3),
            ("tester1", "SAME_STORY", 2),
            ("tester2", "EVOLVED", 1),
        ]
    )
    assert out["tester1"] == {"labels": {"DISTINCT": 3, "SAME_STORY": 2}, "total": 5}
    assert out["tester2"] == {"labels": {"EVOLVED": 1}, "total": 1}


def test_per_annotator_is_empty_without_labels():
    assert _per_annotator([]) == {}


def test_unknown_label_is_rejected():
    with pytest.raises(ValidationError):
        EvalLabelBody.model_validate({"pair_id": PAIR_ID, "annotator": "tester", "label": "MAYBE"})


def test_annotator_must_be_non_empty():
    with pytest.raises(ValidationError):
        EvalLabelBody.model_validate({"pair_id": PAIR_ID, "annotator": "", "label": "DISTINCT"})


@pytest.mark.asyncio
async def test_next_requires_authentication(async_client):
    resp = await async_client.get("/api/v1/eval/next?annotator=forged-human")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_label_rejects_non_uuid_pair_id(signed_in):
    async_client, _ = signed_in
    resp = await async_client.post(
        "/api/v1/eval/label",
        json={"pair_id": "x" * 32, "annotator": "tester", "label": "DISTINCT"},
    )
    assert resp.status_code == 422


def test_eval_routes_are_mounted():
    """The public API surface is asserted through the OpenAPI schema.

    `app.routes` is deliberately not used: FastAPI >= 0.14x / Starlette 1.x
    keeps included routers as nested mounts instead of flattening them, so
    route objects no longer carry a `.path` for every endpoint.
    """
    from app.main import app

    paths = set(app.openapi()["paths"])
    for path in ("/api/v1/eval/next", "/api/v1/eval/label", "/api/v1/eval/progress", "/api/v1/eval/report", "/api/v1/eval/export", "/api/v1/eval/splits", "/api/v1/eval/rubric", "/api/v1/eval/review/next"):
        assert path in paths


@pytest_asyncio.fixture
async def signed_in(async_client, db_session):
    user = await seed_user(db_session)
    await db_session.commit()
    app.dependency_overrides[get_current_user] = lambda: user
    yield async_client, user
    app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.parametrize("extra", [
    {"notes": "x" * 2001},
    {"mutation_types": ["x"] * 17},
    {"mutation_types": ["x" * 65]},
    {"mutation_types": [" "]},
    {"mutation_types": [123]},
    {"origin": "human"},
    {"annotator_user_id": PAIR_ID},
])
def test_label_bounds_and_provenance_are_enforced(extra):
    with pytest.raises(ValidationError):
        EvalLabelBody.model_validate({"pair_id": PAIR_ID, "label": "EVOLVED", **extra})


def test_original_label_body_and_optional_annotations_are_compatible():
    body = EvalLabelBody(pair_id=uuid.UUID(PAIR_ID), label="EVOLVED")
    assert body.model_dump()["annotator"] is None and body.mutation_types is None
    body = EvalLabelBody(pair_id=uuid.UUID(PAIR_ID), label="EVOLVED", mutation_types=["NUMERIC_DRIFT", " NUMERIC_DRIFT ", "HEDGING_SHIFT"], notes="Manual comparison")
    assert body.mutation_types == ["HEDGING_SHIFT", "NUMERIC_DRIFT"]


@pytest.mark.asyncio
async def test_every_eval_read_requires_authentication(async_client):
    for url in ("/next", "/progress", "/report", "/export?dataset_version=v1", "/rubric", "/review/next"):
        response = await async_client.get(f"/api/v1/eval{url}")
        assert response.status_code == 401
    response = await async_client.post("/api/v1/eval/label", json={"pair_id": PAIR_ID, "label": "DISTINCT", "annotator": "forged"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_label_binds_identity_and_retains_legacy_and_revisions(signed_in, db_session):
    client, user = signed_in
    pair, _, _ = await seed_pair(db_session)
    legacy = EvalPairLabel(pair_id=pair.id, annotator=f"user:{user.id}", label="DISTINCT")
    db_session.add(legacy)
    await db_session.commit()
    response = await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "EVOLVED", "annotator": "user:spoofed", "mutation_types": ["NUMERIC_DRIFT"], "notes": "Check the number"})
    assert response.status_code == 201
    assert response.json()["annotator"] == f"user:{user.id}"
    assert response.json()["origin"] == "human"
    votes = (await db_session.execute(select(EvalPairLabel).order_by(EvalPairLabel.origin))).scalars().all()
    assert len(votes) == 2
    human = next(v for v in votes if v.origin == "human")
    original_created = iso(human.created_at)
    assert human.annotator_user_id == user.id
    assert legacy.origin == "legacy" and legacy.label == "DISTINCT"
    response = await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "SAME_STORY"})
    assert response.status_code == 201
    await db_session.refresh(human)
    assert iso(human.created_at) == original_created
    assert human.notes == "Check the number" and human.mutation_types == ["NUMERIC_DRIFT"]
    assert len(human.history) == 1 and human.history[0]["label"] == "EVOLVED"
    assert "history" not in human.history[0]
    assert len((await db_session.execute(select(EvalPairLabel))).scalars().all()) == 2
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "SAME_STORY"})
    await db_session.refresh(human)
    assert len(human.history) == 1  # Idempotent retry is not a new independent vote.


@pytest.mark.asyncio
async def test_label_unknown_pair_and_spoofed_origin_are_rejected(signed_in):
    client, _ = signed_in
    response = await client.post("/api/v1/eval/label", json={"pair_id": PAIR_ID, "label": "DISTINCT"})
    assert response.status_code == 404
    response = await client.post("/api/v1/eval/label", json={"pair_id": PAIR_ID, "label": "DISTINCT", "origin": "automatic"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_next_is_blinded_and_legacy_does_not_hide_work(signed_in, db_session):
    client, user = signed_in
    pair, ca, cb = await seed_pair(db_session, sim=0.938271)
    db_session.add(EvalPairLabel(pair_id=pair.id, annotator=f"user:{user.id}", label="SAME_STORY", origin="legacy"))
    await db_session.commit()
    response = await client.get("/api/v1/eval/next?annotator=another-user")
    assert response.status_code == 200
    payload = response.json()["pair"]
    assert payload == {"pair_id": str(pair.id), "a": {"text": ca.claim_text, "domain": "a.example"}, "b": {"text": cb.claim_text, "domain": "b.example"}}
    assert set(payload) == {"pair_id", "a", "b"}
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "EVOLVED"})
    assert (await client.get("/api/v1/eval/next")).json() == {"done": True, "pair": None}


@pytest.mark.asyncio
async def test_uncertainty_train_only_and_held_out_coverage_fixed(signed_in, db_session):
    client, _ = signed_in
    first, _, _ = await seed_pair(db_session, sim=0.95, split="train", event="train-first", sampled_at=datetime(2026, 9, 1, tzinfo=UTC))
    near, _, _ = await seed_pair(db_session, sim=0.601, split="train", event="train-near", sampled_at=datetime(2026, 9, 2, tzinfo=UTC))
    held_first, _, _ = await seed_pair(db_session, sim=0.95, split="test", event="held-first", sampled_at=datetime(2026, 9, 1, tzinfo=UTC))
    await seed_pair(db_session, sim=0.601, split="test", event="held-near", sampled_at=datetime(2026, 9, 2, tzinfo=UTC))
    await db_session.commit()
    assert (await client.get("/api/v1/eval/next?split=train&strategy=coverage")).json()["pair"]["pair_id"] == str(first.id)
    assert (await client.get("/api/v1/eval/next?split=train&strategy=uncertainty")).json()["pair"]["pair_id"] == str(near.id)
    assert (await client.get("/api/v1/eval/next?split=test&strategy=coverage")).json()["pair"]["pair_id"] == str(held_first.id)
    for split in ("unassigned", "dev", "test"):
        response = await client.get(f"/api/v1/eval/next?split={split}&strategy=uncertainty")
        assert response.status_code == 422 and "split=train" in response.json()["detail"]
    assert (await client.get("/api/v1/eval/next?strategy=not-a-strategy")).status_code == 422


@pytest.mark.asyncio
async def test_progress_counts_humans_and_all_origins_separately(signed_in, db_session):
    client, user = signed_in
    pair, _, _ = await seed_pair(db_session)
    for origin in ("legacy", "automatic", "test"):
        for voter in ("machine-a", "machine-b"):
            db_session.add(EvalPairLabel(pair_id=pair.id, annotator=voter, label="SAME_STORY", origin=origin))
    await db_session.commit()
    initial = (await client.get("/api/v1/eval/progress")).json()
    assert initial["human_votes"] == initial["labeled_votes"] == 0
    assert initial["total_votes_all_origins"] == 6
    assert initial["consensus_pairs"] == 0 and not initial["complete"]
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "EVOLVED"})
    second = await seed_user(db_session, index=2)
    await db_session.commit()
    app.dependency_overrides[get_current_user] = lambda: second
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "EVOLVED", "annotator": f"user:{user.id}"})
    progress = (await client.get("/api/v1/eval/progress")).json()
    assert progress["human_votes"] == progress["labeled_votes"] == 2
    assert progress["human_labeled_pairs"] == progress["consensus_pairs"] == 1
    assert progress["complete"] and progress["total_votes_all_origins"] == 8
    assert all(progress["origins"][origin]["votes"] == 2 for origin in ("human", "legacy", "automatic", "test"))
    assert all(row["origin"] == "human" for row in progress["inter_annotator"])
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "DISTINCT"})
    unresolved = (await client.get("/api/v1/eval/progress")).json()
    assert unresolved["disagreement_pairs"] == 1 and unresolved["consensus_pairs"] == 0
    assert not unresolved["complete"]


@pytest.mark.asyncio
async def test_admin_split_assignment_is_atomic_complete_and_immutable(signed_in, db_session):
    client, user = signed_in
    first, shared, _ = await seed_pair(db_session)
    second, _, _ = await seed_pair(db_session, claim_a=shared)
    await db_session.commit()
    assignment = {"pair_id": str(first.id), "event_group": "event-a", "split": "train"}
    assert (await client.post("/api/v1/eval/splits", json={"assignments": [assignment]})).status_code == 403
    user.role = RoleEnum.admin
    await db_session.commit()
    response = await client.post("/api/v1/eval/splits", json={"assignments": [assignment]})
    assert response.status_code == 409 and "entire connected" in response.json()["detail"]
    await db_session.refresh(first)
    assert first.split == "unassigned"
    changes = [assignment, {**assignment, "pair_id": str(second.id)}]
    response = await client.post("/api/v1/eval/splits", json={"assignments": changes})
    assert response.status_code == 200 and response.json() == {"assigned": 2, "unchanged": 0, "frozen": True}
    assert (await client.post("/api/v1/eval/splits", json={"assignments": changes})).json()["unchanged"] == 2
    response = await client.post("/api/v1/eval/splits", json={"assignments": [{**assignment, "split": "test"}]})
    assert response.status_code == 409 and "frozen" in response.json()["detail"]


@pytest.mark.asyncio
async def test_admin_export_actual_records_deterministic_and_publication_checks(signed_in, db_session, monkeypatch):
    from app.models.claim import ClaimEntity
    from tests.conftest import TestingSessionLocal

    monkeypatch.setattr("app.db.postgres.async_session_maker", TestingSessionLocal)
    client, user = signed_in
    pair, ca, cb = await seed_pair(db_session, sim=0.812345, text_a="Atlas reported 10 flights", text_b="Atlas reported 12 flights")
    other = await seed_user(db_session, index=2)
    db_session.add_all([
        ClaimEntity(claim_id=ca.id, entity_text="Atlas", entity_type="ORG"),
        ClaimEntity(claim_id=cb.id, entity_text="Atlas", entity_type="ORG"),
        EvalPairLabel(pair_id=pair.id, annotator="old-machine", origin="legacy", label="DISTINCT"),
        EvalPairLabel(pair_id=pair.id, annotator="automatic-signal", origin="automatic", label="DISTINCT"),
    ])
    await db_session.commit()
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "EVOLVED", "mutation_types": ["NUMERIC_DRIFT"], "notes": "Human observation"})
    app.dependency_overrides[get_current_user] = lambda: other
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "EVOLVED", "mutation_types": ["NUMERIC_DRIFT"]})
    assert (await client.get("/api/v1/eval/export?dataset_version=v1")).status_code == 403
    app.dependency_overrides[get_current_user] = lambda: user
    user.role = RoleEnum.admin
    await db_session.commit()
    first = await client.get("/api/v1/eval/export?dataset_version=v1")
    second = await client.get("/api/v1/eval/export?dataset_version=v1")
    assert first.status_code == 200 and first.json() == second.json()
    record = first.json()["records"][0]
    assert record["text_a"] == ca.claim_text and record["text_b"] == cb.claim_text
    assert record["cosine_score"] == 0.812345
    assert record["features"]["entity_jaccard"] == 1.0
    assert record["claim_a_id"] == str(ca.id) and record["article_b_id"] == str(cb.article_id)
    assert record["gold_label"] == "EVOLVED" and record["gold_mutation_types"] == ["NUMERIC_DRIFT"]
    assert record["published_at_a"] == "2026-09-01T00:00:00+00:00"
    assert {label["origin"] for label in record["labels"]} == {"human", "legacy", "automatic"}
    assert len(record["human_labels"]) == 2
    assert not first.json()["manifest"]["publication_ready"]
    response = await client.get("/api/v1/eval/export?dataset_version=v1&publication=true")
    assert response.status_code == 422 and "unassigned" in response.json()["detail"]
    response = await client.post("/api/v1/eval/splits", json={"assignments": [{"pair_id": str(pair.id), "event_group": "event-a", "split": "train"}]})
    assert response.status_code == 200
    assert (await client.get("/api/v1/eval/export?dataset_version=v1&publication=true")).status_code == 200


@pytest.mark.asyncio
async def test_real_bearer_identity_not_client_identity(async_client, db_session):
    pair, _, _ = await seed_pair(db_session)
    await db_session.commit()
    registration = await async_client.post("/api/v1/auth/register", json={"email": "eval-real-bearer@example.com", "password": "password123"})
    assert registration.status_code == 201
    token = registration.json()["access_token"]
    me = await async_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    response = await async_client.post("/api/v1/eval/label", headers={"Authorization": f"Bearer {token}"}, json={"pair_id": str(pair.id), "label": "DISTINCT", "annotator": "untrusted-client"})
    assert response.status_code == 201
    assert response.json()["annotator"] == f"user:{me.json()['id']}"


@pytest.mark.asyncio
async def test_span_annotations_are_exact_versioned_and_keep_revisions(signed_in, db_session):
    client, _ = signed_in
    pair, _, _ = await seed_pair(db_session, text_a="🚀 Atlas has 10 flights", text_b="🚀 Atlas has 12 flights")
    await db_session.commit()
    span = {"side": "a", "start": 12, "end": 14, "text": "10", "mutation_type": "NUMERIC_DRIFT"}
    body = {"pair_id": str(pair.id), "label": "EVOLVED", "mutation_types": ["NUMERIC_DRIFT"], "context_relation": "INSUFFICIENT_CONTEXT", "changed_spans": [span, span]}
    recorded = await client.post("/api/v1/eval/label", json=body)
    assert recorded.status_code == 201, recorded.text
    assert recorded.json()["rubric_version"] == "claim-pairs-v2"
    assert recorded.json()["changed_spans"] == [span]
    invalid = await client.post("/api/v1/eval/label", json={**body, "changed_spans": [{**span, "start": 13, "end": 15}]})
    assert invalid.status_code == 422  # UTF-16 offset is invalid after the emoji.
    inconsistent = await client.post("/api/v1/eval/label", json={**body, "mutation_types": []})
    assert inconsistent.status_code == 422
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "EVOLVED", "notes": "Reviewed independently"})
    vote = (await db_session.execute(select(EvalPairLabel))).scalar_one()
    assert vote.provenance["changed_spans"] == [span]
    assert len(vote.history) == 1
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "EVOLVED", "changed_spans": []})
    await db_session.refresh(vote)
    assert vote.provenance["changed_spans"] == []
    assert vote.history[-1]["provenance"]["changed_spans"] == [span]
    rubric = await client.get("/api/v1/eval/rubric")
    assert rubric.status_code == 200 and rubric.json()["version"] == "claim-pairs-v2"


@pytest.mark.asyncio
async def test_disagreement_queue_is_blinded_independent_and_never_overrides_dissent(signed_in, db_session):
    client, _user = signed_in
    pair, _, _ = await seed_pair(db_session)
    other = await seed_user(db_session, index=2)
    reviewer = await seed_user(db_session, index=3, role=RoleEnum.admin)
    await db_session.commit()
    assert (await client.get("/api/v1/eval/review/next")).status_code == 403
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "EVOLVED"})
    app.dependency_overrides[get_current_user] = lambda: other
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "DISTINCT"})
    app.dependency_overrides[get_current_user] = lambda: reviewer
    selected = await client.get("/api/v1/eval/review/next")
    assert selected.status_code == 200
    assert selected.json()["pair"]["pair_id"] == str(pair.id)
    assert set(selected.json()["pair"]) == {"pair_id", "a", "b"}
    assert not any(cue in selected.text for cue in ("sim_score", "annotator", "EVOLVED", "DISTINCT"))
    await client.post("/api/v1/eval/label", json={"pair_id": str(pair.id), "label": "EVOLVED"})
    assert (await client.get("/api/v1/eval/review/next")).json() == {"done": True, "pair": None}
    progress = (await client.get("/api/v1/eval/progress")).json()
    assert progress["disagreement_pairs"] == 1 and progress["consensus_pairs"] == 0
    votes = (await db_session.execute(select(EvalPairLabel))).scalars().all()
    assert len(votes) == 3 and any(v.annotator_user_id == other.id and v.label == "DISTINCT" for v in votes)


@pytest.mark.asyncio
async def test_disagreement_queue_ignores_unverified_identity(signed_in, db_session):
    client, reviewer = signed_in
    reviewer.role = RoleEnum.admin
    first = await seed_user(db_session, index=2)
    second = await seed_user(db_session, index=3)
    pair, _, _ = await seed_pair(db_session)
    db_session.add_all([
        EvalPairLabel(pair_id=pair.id, origin="human", annotator_user_id=first.id, annotator=f"user:{first.id}", label="EVOLVED"),
        EvalPairLabel(pair_id=pair.id, origin="human", annotator_user_id=second.id, annotator="forged-identity", label="DISTINCT"),
    ])
    await db_session.commit()
    assert (await client.get("/api/v1/eval/review/next")).json() == {"done": True, "pair": None}
