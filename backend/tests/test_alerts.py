"""Tests for the early-warning alert engine (pure logic + endpoint)."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.claim import Claim
from app.models.evolution import Alert, ClaimRelationship, RelationshipTypeEnum
from app.services.alerts import build_alerts, evaluate_alerts
from app.services.evolution.text_changes import analyze_text_change

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def _recent(n: int) -> list[datetime]:
    return [NOW - timedelta(hours=2) for _ in range(n)]


def _prior(n: int) -> list[datetime]:
    return [NOW - timedelta(hours=30) for _ in range(n)]


def test_quiet_when_nothing_anomalous():
    out = build_alerts(published_ats=[], disputed_ats=[], mutation_ats=[], now=NOW)
    assert [a["type"] for a in out["alerts"]] == ["QUIET"]
    assert out["stats"]["articles_24h"] == 0


def test_ingestion_spike():
    out = build_alerts(
        published_ats=_recent(6) + _prior(1), disputed_ats=[], mutation_ats=[], now=NOW
    )
    spike = [a for a in out["alerts"] if a["type"] == "INGESTION_SPIKE"]
    assert spike and spike[0]["severity"] == "warn"


def test_no_spike_when_baseline_matches():
    out = build_alerts(
        published_ats=_recent(6) + _prior(6), disputed_ats=[], mutation_ats=[], now=NOW
    )
    assert not [a for a in out["alerts"] if a["type"] == "INGESTION_SPIKE"]


def test_disputed_surge():
    out = build_alerts(
        published_ats=[], disputed_ats=_recent(3), mutation_ats=[], now=NOW
    )
    surge = [a for a in out["alerts"] if a["type"] == "DISPUTED_SURGE"]
    assert surge and surge[0]["severity"] == "warn"


def test_mutation_surge_is_critical():
    out = build_alerts(
        published_ats=[], disputed_ats=[], mutation_ats=_recent(4), now=NOW
    )
    surge = [a for a in out["alerts"] if a["type"] == "MUTATION_SURGE"]
    assert surge and surge[0]["severity"] == "critical"


def test_naive_datetimes_treated_as_utc():
    out = build_alerts(
        published_ats=[datetime(2026, 9, 16, 10, 0, tzinfo=UTC).replace(tzinfo=None)],
        disputed_ats=[],
        mutation_ats=[],
        now=NOW,
    )
    assert out["stats"]["articles_24h"] == 1


def test_explicit_utc_offset_matches_now():
    from datetime import timedelta, timezone

    plus5 = timezone(timedelta(hours=5, minutes=30))
    out = build_alerts(
        published_ats=[datetime(2026, 9, 16, 12, 30, tzinfo=plus5)],
        disputed_ats=[],
        mutation_ats=[],
        now=NOW,
    )
    # 12:30 +05:30 == 07:00 UTC — still inside the 24h window
    assert out["stats"]["articles_24h"] == 1


@pytest.mark.asyncio
async def test_alerts_endpoint(async_client, db_session, monkeypatch):
    """End-to-end on the SQLite test app with seeded rows."""
    import uuid

    from sqlalchemy import text

    from tests.conftest import TestingSessionLocal

    sid = str(uuid.uuid4())
    async with TestingSessionLocal() as db:
        await db.execute(
            text(
                "INSERT INTO sources (id, name, type, url_or_identifier, is_active) "
                "VALUES (:id, 'test', 'rss', 'https://t.example/rss', 1)"
            ),
            {"id": sid},
        )
        art_ids = []
        for i, hours in enumerate([2, 3, 4, 5, 6, 30]):
            aid = str(uuid.uuid4())
            art_ids.append(aid)
            await db.execute(
                text(
                    "INSERT INTO articles (id, source_id, title, url, content_hash,"
                    " published_at) VALUES (:id, :sid, :title, :url, :ch, :pub)"
                ),
                {
                    "id": aid,
                    "sid": sid,
                    "title": f"t{i}",
                    "url": f"https://t.example/{i}",
                    "ch": f"hash-{i}",
                    "pub": (datetime.now(UTC) - timedelta(hours=hours)).isoformat(),
                },
            )
        # 3 recent DISPUTED claims on the first article
        for i in range(3):
            await db.execute(
                text(
                    "INSERT INTO claims (id, article_id, claim_text, verdict)"
                    " VALUES (:id, :aid, :txt, 'DISPUTED')"
                ),
                {"id": str(uuid.uuid4()), "aid": art_ids[0], "txt": f"claim {i}"},
            )
        await db.commit()

    monkeypatch.setattr("app.db.postgres.async_session_maker", TestingSessionLocal)
    resp = await async_client.get("/api/v1/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data and "stats" in data and "generated_at" in data
    assert data["stats"]["articles_24h"] == 5
    assert data["stats"]["disputed_24h"] == 3
    types = {a["type"] for a in data["alerts"]}
    assert "INGESTION_SPIKE" in types
    assert "DISPUTED_SURGE" in types


def _alert_result(rows=(), existing=None) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(rows)
    result.all.return_value = list(rows)
    result.scalar_one_or_none.return_value = existing
    return result


def _persistent_db(parent_text, child_text, *, stored=False, existing_alert=None) -> tuple[MagicMock, Claim, Claim]:
    parent = Claim(id=uuid.UUID(int=1), claim_text=parent_text)
    child = Claim(id=uuid.UUID(int=2), claim_text=child_text)
    evidence = analyze_text_change(parent_text, child_text) if stored else None
    if evidence is not None:
        evidence["algorithm_version"] = "persisted-test-version"
    edge = ClaimRelationship(
        from_claim_id=child.id, to_claim_id=parent.id,
        relationship_type=RelationshipTypeEnum.EVOLVED_FROM, score=0.95,
        mutation_evidence=evidence,
    )
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[
        _alert_result(), _alert_result([edge]), _alert_result([parent, child]),
        _alert_result(), _alert_result(existing=existing_alert),
    ])
    db.commit = AsyncMock()
    return db, parent, child


@pytest.mark.parametrize(("parent_text", "child_text", "direction"), [
    ("Officials reportedly found 10 people.", "Officials found 12 people.", "lost"),
    ("Officials found 10 people.", "Officials reportedly found 12 people.", "gained"),
])
@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.asyncio
async def test_persistent_mutation_alerts_are_parent_to_child_not_reversed(parent_text, child_text, direction, stored):
    db, parent, child = _persistent_db(parent_text, child_text, stored=stored)
    summary = await evaluate_alerts(db)
    assert summary["alerts_inserted"] == 1
    alert = db.add.call_args.args[0]
    assert alert.kind == "MUTATION"
    assert alert.claim_id == child.id
    assert alert.body == f"\u201c{parent_text}\u201d \u2192 \u201c{child_text}\u201d"
    assert alert.payload["from_claim"] == str(parent.id)
    assert alert.payload["to_claim"] == str(child.id)
    assert alert.payload["parent_claim_id"] == str(parent.id)
    assert alert.payload["child_claim_id"] == str(child.id)
    analysis = alert.payload["analysis"]
    hedge = next(c for c in analysis["changes"] if c["kind"] == "hedge")
    assert hedge["direction"] == direction
    assert analysis["observed_propagation"] is False
    assert alert.payload["observed_propagation"] is False
    if stored:
        assert analysis["algorithm_version"] == "persisted-test-version"
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_reversed_alert_is_refreshed_without_losing_acknowledgement():
    existing = Alert(
        kind="MUTATION", severity="INFO", subject_key="edge:test",
        title="Old reversed title", body="Newer to older", claim_id=uuid.UUID(int=1),
        payload={"from_claim": str(uuid.UUID(int=2)), "to_claim": str(uuid.UUID(int=1))},
        acknowledged_at=NOW,
    )
    db, parent, child = _persistent_db("Officials reportedly found 10 people.", "Officials found 12 people.", existing_alert=existing)
    summary = await evaluate_alerts(db)
    assert summary["alerts_refreshed"] == 1
    assert summary["alerts_inserted"] == 0
    db.add.assert_not_called()
    assert existing.acknowledged_at == NOW
    assert existing.claim_id == child.id
    assert existing.payload is not None
    assert existing.payload["from_claim"] == str(parent.id)
    assert existing.payload["to_claim"] == str(child.id)
    assert existing.severity == "WARNING"


@pytest.mark.asyncio
async def test_legacy_duplicate_edge_does_not_raise_mutation_alert():
    db, _, _ = _persistent_db("Officials found 10 people.", "Officials found 10 people.")
    summary = await evaluate_alerts(db)
    assert summary["conditions_detected"] == 0
    db.add.assert_not_called()
