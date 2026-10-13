from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user, get_db_session, require_role
from app.models.evolution import ClaimClusterRun
from app.models.pipeline import JobTypeEnum
from app.models.user import RoleEnum, User
from app.services.evolution.neo4j_writer import Neo4jWriter
from app.services.evolution.orchestrator import EvolutionOrchestrator
from app.services.pipeline.jobs import JobAlreadyRunning, run_job

router = APIRouter()


@router.post("/graph/sync", dependencies=[Depends(require_role(RoleEnum.admin))])
async def reconcile_graph(db: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Rebuild the bounded graph projection from authoritative persisted records."""
    result = await Neo4jWriter().sync_persisted_graph(db)
    if not result["success"]:
        raise HTTPException(status_code=503, detail=result)
    return result


@router.post("/trigger")
async def trigger_evolution(
    force: bool = Query(False, description="Bypass debounce guard (minimum-corpus guard is still respected)"),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_role([RoleEnum.admin])),
) -> Any:
    orchestrator = EvolutionOrchestrator()
    try:
        async with run_job(
            db, JobTypeEnum.evolution, trigger="manual", user_id=current_user.id
        ) as job:
            result = await orchestrator.run_evolution_cycle(db, force=force)
            job.summary = result if isinstance(result, dict) else {"result": str(result)}
    except JobAlreadyRunning:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="An evolution job is already running."
        )
    return {"message": "Evolution cycle triggered", "job_id": str(job.id), "result": result}


@router.get("/status", dependencies=[Depends(get_current_user)])
async def get_evolution_status(db: AsyncSession = Depends(get_db_session)) -> Any:
    stmt = select(ClaimClusterRun).order_by(ClaimClusterRun.run_at.desc()).limit(1)
    run = await db.scalar(stmt)
    
    if not run:
        return {"status": "no_runs_yet"}
        
    return {
        "status": "success",
        "last_run_id": run.id,
        "last_run_at": run.run_at,
        "claims_in_corpus_at_run": run.claims_in_corpus,
        "algorithm_params": run.algorithm_params
    }


@router.get("/clusters", dependencies=[Depends(get_current_user)])
async def get_clusters(db: AsyncSession = Depends(get_db_session)) -> Any:
    # Fetch the latest run with assignments
    stmt = (
        select(ClaimClusterRun)
        .options(selectinload(ClaimClusterRun.assignments))
        .order_by(ClaimClusterRun.run_at.desc())
        .limit(1)
    )
    run = await db.scalar(stmt)
    
    if not run:
        return {"clusters": []}
        
    # Group assignments by topic_id
    clusters: dict[int, dict[str, Any]] = {}
    for a in run.assignments:
        if a.topic_id not in clusters:
            clusters[a.topic_id] = {
                "topic_id": a.topic_id,
                "label": a.topic_label,
                "keywords": a.topic_keywords,
                "claim_ids": []
            }
        clusters[a.topic_id]["claim_ids"].append(a.claim_id)
        
    return {"clusters": list(clusters.values())}

@router.get("/clusters/{topic_id}/mutation-summary", dependencies=[Depends(get_current_user)])
async def get_cluster_mutation_summary(topic_id: int, db: AsyncSession = Depends(get_db_session)) -> Any:
    """Generates an LLM summary of how claims in this topic cluster mutated over time."""
    from app.models.article import Article
    from app.models.claim import Claim
    from app.services.nlp.llm_client import get_llm_client
    
    # Fetch claims in the topic ordered by time
    stmt = (
        select(ClaimClusterRun)
        .options(selectinload(ClaimClusterRun.assignments))
        .order_by(ClaimClusterRun.run_at.desc())
        .limit(1)
    )
    run = await db.scalar(stmt)
    if not run:
        raise HTTPException(status_code=404, detail="No cluster run found")
        
    claim_ids = [a.claim_id for a in run.assignments if a.topic_id == topic_id]
    if not claim_ids:
        raise HTTPException(status_code=404, detail="Topic not found or empty")
        
    # Get chronological claims
    claims_stmt = (
        select(Claim.claim_text)
        .join(Article, Claim.article_id == Article.id)
        .where(Claim.id.in_(claim_ids))
        .order_by(Article.published_at.asc().nulls_last())
        .limit(50)  # limit context window
    )
    claims_rows = await db.scalars(claims_stmt)
    claims = list(claims_rows.all())
    
    if len(claims) < 2:
        return {"summary": "Insufficient claims to detect a temporal mutation."}
        
    llm = get_llm_client()
    summary = await llm.generate_mutation_summary(claims)
    return {"topic_id": topic_id, "summary": summary}

