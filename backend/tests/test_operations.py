"""Admin-only bounded inspection and recovery preserve active row ownership."""
import uuid
from datetime import UTC, datetime, timedelta

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.main import app
from app.models.article import Article, NLPStatusEnum, ProcessingStatusEnum
from app.models.source import Source, SourceTypeEnum
from app.models.user import RoleEnum, User


async def test_operations_requires_administrator(async_client: AsyncClient) -> None:
    assert (await async_client.get("/api/v1/operations/jobs")).status_code == 401
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uuid.uuid4(), email="reader@example.com", role=RoleEnum.user
    )
    try:
        assert (await async_client.get("/api/v1/operations/metrics")).status_code == 403
        assert (await async_client.post("/api/v1/operations/recover", json={"stage": "nlp"})).status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_user, None)


async def test_recovery_defaults_dry_run_and_never_seizes_live_leases(
    async_client: AsyncClient, db_session: AsyncSession
) -> None:
    source = Source(name="Owned fixture", type=SourceTypeEnum.rss, url_or_identifier="fixture")
    db_session.add(source)
    await db_session.flush()
    now = datetime.now(UTC)
    expired = Article(source_id=source.id, title="Expired", url="https://fixture/expired",
        content_hash="expired", processing_status=ProcessingStatusEnum.processed,
        nlp_status=NLPStatusEnum.processing, claimed_by="expired-worker", attempt_count=1,
        claim_expires_at=now - timedelta(seconds=1))
    active = Article(source_id=source.id, title="Active", url="https://fixture/active",
        content_hash="active", processing_status=ProcessingStatusEnum.processed,
        nlp_status=NLPStatusEnum.processing, claimed_by="live-worker", attempt_count=1,
        claim_expires_at=now + timedelta(hours=1))
    db_session.add_all([expired, active])
    await db_session.commit()
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uuid.uuid4(), email="operator@example.com", role=RoleEnum.admin
    )
    try:
        response = await async_client.post("/api/v1/operations/recover", json={"stage": "nlp"})
        assert response.status_code == 200 and response.json()["dry_run"] is True
        assert response.json()["requeued"] == 1
        await db_session.refresh(expired)
        assert expired.nlp_status == NLPStatusEnum.processing
        response = await async_client.post("/api/v1/operations/recover", json={"stage": "nlp", "dry_run": False})
        assert response.status_code == 200 and response.json()["requeued"] == 1
        await db_session.refresh(expired)
        await db_session.refresh(active)
        assert expired.nlp_status.value == "pending" and expired.claimed_by is None
        assert active.nlp_status == NLPStatusEnum.processing and active.claimed_by == "live-worker"
        metrics = await async_client.get("/api/v1/operations/metrics")
        assert metrics.json()["nlp"] == {"pending": 1, "processing": 1}
        assert metrics.json()["expired_article_leases"] == 0
        assert (await async_client.get("/api/v1/operations/jobs?limit=101")).status_code == 422
    finally:
        app.dependency_overrides.pop(get_current_user, None)
