import asyncio
import logging
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.article import Article, NLPStatusEnum
from app.models.claim import Claim, ClaimEntity
from app.services.nlp.embedding_service import EmbeddingService
from app.services.nlp.entity_extractor import EntityExtractor
from app.services.nlp.llm_client import LLMClient, get_llm_client
from app.services.pipeline.ownership import (
    Stage,
    claim_articles,
    extend_claims,
    finish_article,
    new_owner_token,
    recover_expired,
    release_unfinished,
    sanitize_error,
)

logger = logging.getLogger(__name__)


class NLPSummary(BaseModel):
    total_processed: int = 0
    status_counts: dict[str, int] = Field(default_factory=lambda: {
        NLPStatusEnum.completed.value: 0,
        NLPStatusEnum.skipped.value: 0,
        NLPStatusEnum.failed.value: 0,
    })
    claims_extracted: int = 0
    llm_calls_made: int = 0
    errors: list[str] = Field(default_factory=list)
    lost_ownership: int = 0
    recovered: dict[str, int] = Field(default_factory=dict)


class NLPOrchestrator:
    def __init__(self, llm_client: LLMClient | None = None):
        self.settings = get_settings()
        self.max_articles = getattr(self.settings, 'NLP_MAX_ARTICLES_PER_CYCLE', 10)
        self.llm_client = llm_client or get_llm_client()

    async def run_nlp_cycle(self, db: AsyncSession, *, chain_evolution: bool = True) -> NLPSummary:
        summary = NLPSummary()
        owner = new_owner_token("nlp")

        recovery = await recover_expired(db, Stage.nlp)
        summary.recovered = {"requeued": recovery.requeued, "failed": recovery.failed}

        # Atomic claim: concurrent workers get disjoint articles.
        claimed_ids = await claim_articles(db, Stage.nlp, owner, self.max_articles)
        if not claimed_ids:
            logger.info("No articles pending NLP processing.")
            return summary

        logger.info(
            "Starting NLP cycle. Claimed %d pending articles (cap: %d).",
            len(claimed_ids),
            self.max_articles,
        )
        rows = (
            await db.execute(
                select(Article.id, Article.title, Article.cleaned_content, Article.content).where(
                    Article.id.in_(claimed_ids)
                )
            )
        ).all()
        texts = {row.id: (row.cleaned_content or row.content) for row in rows}
        titles = {row.id: row.title for row in rows}

        try:
            for index, article_id in enumerate(claimed_ids):
                if index:
                    # Heartbeat between (potentially slow) LLM calls.
                    await extend_claims(db, Stage.nlp, owner)
                await self._process_one(
                    db, article_id, texts.get(article_id), owner, summary,
                    title=titles.get(article_id, ""),
                )
        finally:
            # Anything still claimed (e.g. cancellation) goes back to the queue.
            released = await release_unfinished(db, Stage.nlp, owner)
            if released:
                logger.warning("Released %d unfinished NLP claims back to pending", released)

        logger.info(
            "NLP cycle completed. Processed %d articles. Extracted %d claims.",
            summary.total_processed,
            summary.claims_extracted,
        )

        if chain_evolution:
            # Trigger evolution cycle (it has its own guards)
            try:
                from app.services.evolution.orchestrator import EvolutionOrchestrator
                evo_orch = EvolutionOrchestrator()
                evo_summary = await evo_orch.run_evolution_cycle(db)
                logger.info("Auto-triggered evolution cycle result: %s", evo_summary)
            except Exception:
                logger.exception("Error during auto-triggered evolution cycle")

        return summary

    async def _process_one(
        self,
        db: AsyncSession,
        article_id: uuid.UUID,
        text_to_process: str | None,
        owner: str,
        summary: NLPSummary,
        *,
        title: str = "",
    ) -> None:
        try:
            new_claims: list[Claim] = []
            article_embedding: list[float] | None = None
            if text_to_process:
                extracted_claims_data = await self.llm_client.extract_claims(text_to_process)
                summary.llm_calls_made += 1
                for claim_data in extracted_claims_data:
                    # CPU-bound inference (and a possible first-use model
                    # load) runs in a worker thread, not on the event loop.
                    entities = await asyncio.to_thread(
                        EntityExtractor.extract_entities, claim_data.claim_text, allow_load=True
                    )
                    embedding = await asyncio.to_thread(
                        EmbeddingService.generate_embedding, claim_data.claim_text, allow_load=True
                    )
                    claim_model = Claim(
                        article_id=article_id,
                        claim_text=claim_data.claim_text,
                        confidence=claim_data.confidence,
                        embedding=embedding,
                        extracted_at=datetime.now(UTC),
                    )
                    for ent in entities:
                        claim_model.entities.append(ClaimEntity(
                            entity_text=ent.entity_text,
                            entity_type=ent.entity_type,
                            start_char=ent.start_char,
                            end_char=ent.end_char,
                        ))
                    new_claims.append(claim_model)

                # Corpus search must index processed articles even when no
                # check-worthy claims were found (e.g. an opinion piece).
                article_embedding = await asyncio.to_thread(
                    EmbeddingService.generate_document_embedding,
                    f"{title}. {text_to_process}" if title else text_to_process,
                    allow_load=True,
                )

            # Fenced status write first, then claims, in ONE transaction: if
            # ownership was lost the rollback discards these claims too.
            won = await finish_article(
                db, Stage.nlp, article_id, owner,
                status=NLPStatusEnum.completed, commit=False,
                values={"embedding": article_embedding} if article_embedding is not None else None,
            )
            if not won:
                summary.lost_ownership += 1
                return
            if new_claims:
                db.add_all(new_claims)
            await db.commit()
            summary.claims_extracted += len(new_claims)
            summary.status_counts[NLPStatusEnum.completed.value] += 1
            summary.total_processed += 1

        except Exception as e:
            logger.exception("Error during NLP processing for article %s", article_id)
            await db.rollback()
            error = sanitize_error(e)
            try:
                await finish_article(
                    db, Stage.nlp, article_id, owner,
                    status=NLPStatusEnum.failed, error=error,
                )
            except Exception as fallback_e:
                logger.error(
                    "Failed to mark article %s as failed: %s", article_id, sanitize_error(fallback_e)
                )
                await db.rollback()
            summary.status_counts[NLPStatusEnum.failed.value] += 1
            summary.errors.append(f"Article {article_id}: {error}")
            summary.total_processed += 1
