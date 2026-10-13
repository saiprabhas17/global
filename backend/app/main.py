from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.v1 import (
    alerts,
    auth,
    collection,
    corpus,
    dashboard,
    eval,
    evolution,
    graph,
    health,
    lineage,
    nlp,
    operations,
    preprocessing,
    verdicts,
)
from app.core import scheduler
from app.core.config import get_settings
from app.core.observability import install_request_context
from app.core.ops_security import install_rate_limiter, install_security_headers
from app.db.postgres import engine
from app.services.pipeline.jobs import JobLeaseLost
from app.services.seeder import seed_sources


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Models are lazy-loaded when needed (to avoid OOM on limited memory)
    # Startup
    settings = get_settings()

    # Auto-enable pgvector extension (idempotent)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    await seed_sources()
    if settings.ENABLE_SCHEDULER:
        scheduler.start_scheduler()
    yield
    # Shutdown
    if settings.ENABLE_SCHEDULER:
        scheduler.stop_scheduler()

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="GMEE API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_security_headers(app)
    install_rate_limiter(app)
    install_request_context(app)

    @app.exception_handler(JobLeaseLost)
    async def job_lease_lost(request: Request, exc: JobLeaseLost) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    api_router = APIRouter()
    api_router.include_router(health.router, prefix="/health", tags=["health"])
    api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
    api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
    api_router.include_router(collection.router, prefix="/collection", tags=["Collection"])
    api_router.include_router(preprocessing.router, prefix="/preprocessing", tags=["Preprocessing"])
    api_router.include_router(nlp.router, prefix="/nlp", tags=["NLP"])
    api_router.include_router(operations.router, prefix="/operations", tags=["Operations"])
    api_router.include_router(evolution.router, prefix="/evolution", tags=["Evolution"])
    api_router.include_router(corpus.router, prefix="/corpus", tags=["Corpus"])
    api_router.include_router(verdicts.router, prefix="/verdicts", tags=["Verdicts"])
    api_router.include_router(graph.router, prefix="/graph", tags=["Graph"])
    api_router.include_router(lineage.router, prefix="/graph", tags=["Graph"])
    api_router.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
    api_router.include_router(eval.router, prefix="/eval", tags=["Evaluation"])

    app.include_router(api_router, prefix="/api/v1")

    return app

app = create_app()
