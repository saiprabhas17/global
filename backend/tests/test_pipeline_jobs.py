"""Durable job renewal stops a worker whose fencing lease is lost."""
import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.pipeline import JobStatusEnum, JobTypeEnum
from app.services.pipeline import jobs


async def test_long_job_renews_and_finishes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = jobs.JobHandle(uuid.uuid4(), "nlp", "test-owner")
    renewal = asyncio.Event()

    async def heartbeat(*args: object) -> bool:
        renewal.set()
        return True

    finish = AsyncMock(return_value=True)
    monkeypatch.setattr(jobs, "start_job", AsyncMock(return_value=handle))
    monkeypatch.setattr(jobs, "heartbeat", AsyncMock(side_effect=heartbeat))
    monkeypatch.setattr(jobs, "finish_job", finish)
    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(PIPELINE_LEASE_SECONDS=1))
    async with jobs.run_job(MagicMock(), JobTypeEnum.nlp, trigger="manual") as job:
        await asyncio.wait_for(renewal.wait(), timeout=3)
        job.summary = {"processed": 1}
    finish.assert_awaited_once()
    assert finish.call_args.kwargs["status"] == JobStatusEnum.succeeded
    assert finish.call_args.kwargs["summary"] == {"processed": 1}


async def test_lost_job_lease_cancels_work_and_records_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = jobs.JobHandle(uuid.uuid4(), "nlp", "test-owner")
    finish = AsyncMock(return_value=False)
    monkeypatch.setattr(jobs, "start_job", AsyncMock(return_value=handle))
    monkeypatch.setattr(jobs, "heartbeat", AsyncMock(return_value=False))
    monkeypatch.setattr(jobs, "finish_job", finish)
    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(PIPELINE_LEASE_SECONDS=1))
    with pytest.raises(jobs.JobLeaseLost):
        async with jobs.run_job(MagicMock(), JobTypeEnum.nlp, trigger="manual"):
            await asyncio.sleep(5)
    finish.assert_awaited_once()
    assert finish.call_args.kwargs["status"] == JobStatusEnum.failed
