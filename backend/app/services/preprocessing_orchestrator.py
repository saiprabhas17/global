import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.article import Article, ProcessingStatusEnum
from app.services.nlp_orchestrator import NLPOrchestrator
from app.services.pipeline.ownership import (
    Stage,
    claim_articles,
    finish_article,
    new_owner_token,
    recover_expired,
    sanitize_error,
)
from app.services.preprocessing.cleaner import clean_text
from app.services.preprocessing.language_detector import detect_language
from app.services.preprocessing.metadata_extractor import (
    compute_word_count,
    extract_domain,
    extract_or_repair_published_at,
)
from app.services.preprocessing.near_duplicate_detector import NearDuplicateDetector

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingSummary:
    total_processed: int = 0
    status_counts: dict[str, int] = field(default_factory=lambda: {
        ProcessingStatusEnum.processed.value: 0,
        ProcessingStatusEnum.skipped_non_english.value: 0,
        ProcessingStatusEnum.failed.value: 0,
    })
    errors: list[str] = field(default_factory=list)
    lost_ownership: int = 0
    recovered: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _ArticleRow:
    """Immutable snapshot of the inputs preprocessing needs."""

    id: uuid.UUID
    content: str | None
    url: str
    published_at: datetime | None
    raw_metadata: dict[str, Any]
    collected_at: datetime
    canonical_article_id: uuid.UUID | None


class PreprocessingOrchestrator:
    BATCH_SIZE = 100
    # Hard bound on batches per cycle so one call cannot run unboundedly.
    MAX_BATCHES = 50

    async def run_preprocessing_cycle(
        self, db: AsyncSession, *, chain_nlp: bool = True
    ) -> PreprocessingSummary:
        logger.info("Starting preprocessing cycle")
        summary = PreprocessingSummary()
        owner = new_owner_token("prep")

        recovery = await recover_expired(db, Stage.preprocessing)
        summary.recovered = {"requeued": recovery.requeued, "failed": recovery.failed}

        dup_detector = NearDuplicateDetector(db)
        await dup_detector.initialize()

        for _ in range(self.MAX_BATCHES):
            # 1. Atomically claim a batch (no read-then-write race).
            claimed_ids = await claim_articles(db, Stage.preprocessing, owner, self.BATCH_SIZE)
            if not claimed_ids:
                break
            rows = (
                await db.execute(
                    select(
                        Article.id,
                        Article.content,
                        Article.url,
                        Article.published_at,
                        Article.raw_metadata,
                        Article.collected_at,
                        Article.canonical_article_id,
                    ).where(Article.id.in_(claimed_ids))
                )
            ).all()
            by_id = {row.id: _ArticleRow(*row) for row in rows}

            for article_id in claimed_ids:
                article = by_id.get(article_id)
                if article is None:
                    continue
                try:
                    values = self._process(article, dup_detector)
                    status = values.pop("processing_status")
                    if not isinstance(status, ProcessingStatusEnum):
                        raise TypeError("Preprocessing returned an invalid status")
                    won = await finish_article(
                        db, Stage.preprocessing, article_id, owner, status=status, values=values
                    )
                    if not won:
                        summary.lost_ownership += 1
                        continue
                    summary.status_counts[status.value] += 1
                    summary.total_processed += 1

                except Exception as e:
                    logger.exception("Error preprocessing article %s", article_id)
                    await db.rollback()
                    error = sanitize_error(e)
                    try:
                        await finish_article(
                            db,
                            Stage.preprocessing,
                            article_id,
                            owner,
                            status=ProcessingStatusEnum.failed,
                            values={"processed_at": datetime.now(UTC)},
                            error=error,
                        )
                    except Exception as fallback_e:
                        logger.error(
                            "Failed to mark article %s as failed: %s",
                            article_id,
                            sanitize_error(fallback_e),
                        )
                        await db.rollback()

                    summary.status_counts[ProcessingStatusEnum.failed.value] += 1
                    summary.errors.append(f"Article {article_id}: {error}")
                    summary.total_processed += 1

        logger.info("Preprocessing cycle completed. Processed: %d", summary.total_processed)

        if chain_nlp:
            try:
                logger.info("Triggering NLP cycle.")
                nlp_orchestrator = NLPOrchestrator()
                nlp_summary = await nlp_orchestrator.run_nlp_cycle(db)
                logger.info("Triggered NLP cycle. Total Processed: %d", nlp_summary.total_processed)
            except Exception as e:
                logger.error("Error triggering NLP cycle: %s", sanitize_error(e))

        return summary

    @staticmethod
    def _process(article: _ArticleRow, dup_detector: NearDuplicateDetector) -> dict[str, object]:
        """Pure computation of the preprocessing result; no DB writes."""
        cleaned = clean_text(article.content)
        lang = detect_language(cleaned)
        now = datetime.now(UTC)
        # (Cross-Lingual Tracking): Non-English articles are no longer skipped.
        # They proceed to the NLP pipeline where the LLM will extract and translate claims to English.

        canonical_id = None
        if cleaned:
            # A transient snapshot for the in-memory LSH index; never added
            # to the session, so it cannot trigger an unfenced ORM flush.
            snapshot = Article(
                id=article.id,
                cleaned_content=cleaned,
                collected_at=article.collected_at,
                canonical_article_id=article.canonical_article_id,
            )
            canonical_id = dup_detector.find_and_insert(snapshot)

        values: dict[str, object] = {
            "processing_status": ProcessingStatusEnum.processed,
            "cleaned_content": cleaned,
            "language": lang,
            "domain": extract_domain(article.url),
            "word_count": compute_word_count(cleaned),
            "published_at": extract_or_repair_published_at(article.published_at, article.raw_metadata),
            "processed_at": now,
        }
        if canonical_id:
            values["canonical_article_id"] = canonical_id
        return values
