"""Bounded administrator inspection and recovery of durable pipeline work."""
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session, require_role
from app.models.article import Article
from app.models.pipeline import JobTypeEnum, PipelineJob
from app.models.user import RoleEnum
from app.services.pipeline.jobs import job_to_dict
from app.services.pipeline.ownership import Stage, recover_expired

router = APIRouter(dependencies=[Depends(require_role(RoleEnum.admin))])


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    job_type: JobTypeEnum | None = None,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    statement = select(PipelineJob)
    if job_type is not None:
        statement = statement.where(PipelineJob.job_type == job_type.value)
    result = await db.execute(
        statement.order_by(PipelineJob.started_at.desc(), PipelineJob.id.desc()).limit(limit)
    )
    return {"items": [job_to_dict(job) for job in result.scalars()]}


class RecoveryRequest(BaseModel):
    stage: Stage
    dry_run: bool = True


@router.post("/recover")
async def recover_articles(
    body: RecoveryRequest, db: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    # Never seize active leases or legacy unleased rows through this endpoint.
    # The latter require an operator to stop all workers first.
    result = await recover_expired(db, body.stage, dry_run=body.dry_run)
    return result.as_dict()


@router.get("/metrics")
async def pipeline_metrics(db: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    processing = await db.execute(
        select(Article.processing_status, func.count()).group_by(Article.processing_status)
    )
    nlp = await db.execute(select(Article.nlp_status, func.count()).group_by(Article.nlp_status))
    jobs = await db.execute(select(PipelineJob.status, func.count()).group_by(PipelineJob.status))
    expired = await db.scalar(select(func.count()).select_from(Article).where(
        Article.claimed_by.is_not(None), Article.claim_expires_at < datetime.now(UTC)
    ))
    return {
        "preprocessing": {state.value: count for state, count in processing},
        "nlp": {state.value: count for state, count in nlp},
        "jobs": {state: count for state, count in jobs},
        "expired_article_leases": int(expired or 0),
    }
