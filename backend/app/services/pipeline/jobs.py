"""Durable job records for pipeline runs (replaces worker-local status globals).

Usage::

    async with run_job(db, JobTypeEnum.nlp, trigger="manual", user_id=u.id) as job:
        summary = await orchestrator.run_nlp_cycle(db)
        job.summary = summary.model_dump()

* At most one ``running`` row per job type (partial unique index). A second
  concurrent start raises :class:`JobAlreadyRunning` (HTTP 409 at the API).
* Expired ``running`` rows (crashed worker) are marked ``abandoned`` before a
  new start, so a crash cannot wedge a job type forever.
* Terminal writes are fenced on ``owner`` so an abandoned worker that wakes up
  later cannot overwrite the state of a newer run.
* Job bookkeeping uses its own short-lived session so orchestrator
  rollbacks never discard job state.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.models.pipeline import JobStatusEnum, JobTypeEnum, PipelineJob
from app.services.pipeline.ownership import new_owner_token, sanitize_error

logger = logging.getLogger(__name__)


class JobAlreadyRunning(RuntimeError):
    def __init__(self, job_type: str) -> None:
        super().__init__(f"a {job_type} job is already running")
        self.job_type = job_type


class JobLeaseLost(RuntimeError):
    """Stop work when the durable job lease can no longer be renewed."""


@dataclass
class JobHandle:
    id: uuid.UUID
    job_type: str
    owner: str
    summary: dict[str, Any] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(UTC)


def _session_factory(db: AsyncSession) -> async_sessionmaker[AsyncSession]:
    # A sibling session on the same engine: independent transaction.
    return async_sessionmaker(bind=db.bind, expire_on_commit=False, class_=AsyncSession)


async def abandon_expired(db: AsyncSession, job_type: str | None = None) -> int:
    stmt = update(PipelineJob).where(
        PipelineJob.status == JobStatusEnum.running.value,
        PipelineJob.lease_expires_at < _now(),
    )
    if job_type:
        stmt = stmt.where(PipelineJob.job_type == job_type)
    result = await db.execute(
        stmt.values(
            status=JobStatusEnum.abandoned.value,
            finished_at=_now(),
            error="LeaseExpired: worker stopped heartbeating",
        ).execution_options(synchronize_session=False)
    )
    await db.commit()
    return int(result.rowcount or 0)  # type: ignore[attr-defined]


async def start_job(
    db: AsyncSession,
    job_type: JobTypeEnum,
    *,
    trigger: str,
    user_id: uuid.UUID | None = None,
) -> JobHandle:
    lease = get_settings().PIPELINE_LEASE_SECONDS
    owner = new_owner_token(job_type.value)
    async with _session_factory(db)() as jobs_db:
        await abandon_expired(jobs_db, job_type.value)
        job = PipelineJob(
            id=uuid.uuid4(),
            job_type=job_type.value,
            status=JobStatusEnum.running.value,
            trigger=trigger,
            triggered_by=user_id,
            owner=owner,
            started_at=_now(),
            heartbeat_at=_now(),
            lease_expires_at=_now() + timedelta(seconds=lease),
            summary={},
        )
        jobs_db.add(job)
        try:
            await jobs_db.commit()
        except IntegrityError as exc:
            await jobs_db.rollback()
            raise JobAlreadyRunning(job_type.value) from exc
    return JobHandle(id=job.id, job_type=job_type.value, owner=owner)


async def heartbeat(db: AsyncSession, handle: JobHandle) -> bool:
    lease = get_settings().PIPELINE_LEASE_SECONDS
    async with _session_factory(db)() as jobs_db:
        result = await jobs_db.execute(
            update(PipelineJob)
            .where(
                PipelineJob.id == handle.id,
                PipelineJob.owner == handle.owner,
                PipelineJob.status == JobStatusEnum.running.value,
            )
            .values(heartbeat_at=_now(), lease_expires_at=_now() + timedelta(seconds=lease))
            .execution_options(synchronize_session=False)
        )
        await jobs_db.commit()
        return (result.rowcount or 0) == 1  # type: ignore[attr-defined]


async def finish_job(
    db: AsyncSession,
    handle: JobHandle,
    *,
    status: JobStatusEnum,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> bool:
    async with _session_factory(db)() as jobs_db:
        result = await jobs_db.execute(
            update(PipelineJob)
            .where(
                PipelineJob.id == handle.id,
                PipelineJob.owner == handle.owner,
                PipelineJob.status == JobStatusEnum.running.value,
            )
            .values(
                status=status.value,
                finished_at=_now(),
                summary=summary or {},
                error=error,
            )
            .execution_options(synchronize_session=False)
        )
        await jobs_db.commit()
        won = (result.rowcount or 0) == 1  # type: ignore[attr-defined]
    if not won:
        logger.warning("job %s finished after losing its lease; state not overwritten", handle.id)
    return won


@contextlib.asynccontextmanager
async def run_job(
    db: AsyncSession,
    job_type: JobTypeEnum,
    *,
    trigger: str,
    user_id: uuid.UUID | None = None,
) -> AsyncIterator[JobHandle]:
    handle = await start_job(db, job_type, trigger=trigger, user_id=user_id)
    owner_task = asyncio.current_task()
    lost_lease = False

    async def renew_lease() -> None:
        nonlocal lost_lease
        interval = max(1.0, get_settings().PIPELINE_LEASE_SECONDS / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                renewed = await heartbeat(db, handle)
            except Exception:
                logger.exception("Could not renew durable job lease %s", handle.id)
                renewed = False
            if not renewed:
                lost_lease = True
                if owner_task is not None:
                    owner_task.cancel()
                return

    renewer = asyncio.create_task(renew_lease())
    try:
        yield handle
    except BaseException as exc:
        await finish_job(
            db, handle, status=JobStatusEnum.failed, summary=handle.summary, error=sanitize_error(exc)
        )
        if lost_lease:
            raise JobLeaseLost("The pipeline job lost its lease; processing stopped.") from exc
        raise
    else:
        await finish_job(db, handle, status=JobStatusEnum.succeeded, summary=handle.summary)
    finally:
        renewer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await renewer


async def latest_job(db: AsyncSession, job_type: JobTypeEnum) -> PipelineJob | None:
    result = await db.execute(
        select(PipelineJob)
        .where(PipelineJob.job_type == job_type.value)
        .order_by(PipelineJob.started_at.desc())
        .limit(1)
    )
    return result.scalars().first()


def job_to_dict(job: PipelineJob) -> dict[str, Any]:
    def iso(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    return {
        "id": str(job.id),
        "job_type": job.job_type,
        "status": job.status,
        "trigger": job.trigger,
        "triggered_by": str(job.triggered_by) if job.triggered_by else None,
        "started_at": iso(job.started_at),
        "heartbeat_at": iso(job.heartbeat_at),
        "lease_expires_at": iso(job.lease_expires_at),
        "finished_at": iso(job.finished_at),
        "summary": job.summary or {},
        "error": job.error,
    }
