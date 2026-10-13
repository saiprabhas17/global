import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import CursorResult, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.article import Article
from app.models.source import Source, SourceTypeEnum
from app.services.collectors.base import BaseCollector, MissingCredentialsError
from app.services.collectors.news_api import NewsAPICollector
from app.services.collectors.reddit import RedditCollector
from app.services.collectors.rss import RSSCollector
from app.services.pipeline.ownership import sanitize_error
from app.services.preprocessing_orchestrator import PreprocessingOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class SourceCollectionSummary:
    source_name: str
    articles_fetched: int
    articles_inserted: int
    error: str | None


class CollectionOrchestrator:
    def __init__(self) -> None:
        self.collectors: dict[SourceTypeEnum, BaseCollector] = {
            SourceTypeEnum.rss: RSSCollector(),
            SourceTypeEnum.news_api: NewsAPICollector(),
            SourceTypeEnum.reddit: RedditCollector()
        }

    def _generate_content_hash(self, title: str, content: str | None) -> str:
        text = f"{title.strip().lower()}:{content.strip().lower() if content else ''}"
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    async def run_collection_cycle(self, db: AsyncSession) -> list[SourceCollectionSummary]:
        logger.info("Starting collection cycle")
        
        # Load all active sources
        result = await db.execute(select(Source.id).where(Source.is_active == True))
        source_ids = list(result.scalars().all())
        
        summaries = []
        
        for source_id in source_ids:
            # Re-fetch per iteration: a rollback for a failed source expires
            # every ORM object; get() reloads an expired instance safely.
            source = await db.get(Source, source_id)
            if source is None:
                continue
            collector = self.collectors.get(source.type)
            if not collector:
                logger.error(f"No collector configured for source type {source.type}")
                summaries.append(SourceCollectionSummary(
                    source_name=source.name,
                    articles_fetched=0,
                    articles_inserted=0,
                    error=f"No collector for type {source.type}"
                ))
                continue
                
            try:
                # 1. Collect raw articles
                raw_articles = await collector.collect(source)
                
                # 2. Prepare for insert
                articles_to_insert = []
                for raw in raw_articles:
                    articles_to_insert.append({
                        "source_id": source.id,
                        "title": raw.title,
                        "url": raw.url,
                        "content": raw.content,
                        "published_at": raw.published_at,
                        "author": raw.author,
                        "raw_metadata": raw.raw_metadata,
                        "content_hash": self._generate_content_hash(raw.title, raw.content),
                        # collected_at and id are handled by defaults
                    })
                
                inserted_count = 0
                if articles_to_insert:
                    # 3. Bulk insert with ON CONFLICT DO NOTHING on url
                    # We deduplicate primarily by url.
                    if db.get_bind().dialect.name == "sqlite":
                        stmt_sq = sqlite_insert(Article).values(articles_to_insert)
                        res = await db.execute(
                            stmt_sq.on_conflict_do_nothing(index_elements=['url'])
                        )
                    else:
                        stmt_pg = pg_insert(Article).values(articles_to_insert)
                        res = await db.execute(
                            stmt_pg.on_conflict_do_nothing(index_elements=['url'])
                        )

                    if isinstance(res, CursorResult):
                        # Bulk operation may ignore duplicates — count actual inserts.
                        inserted_count = res.rowcount or 0
                    else:
                        inserted_count = 0
                
                # 4. Update last_collected_at
                source.last_collected_at = datetime.now(UTC)
                await db.commit()
                
                summaries.append(SourceCollectionSummary(
                    source_name=source.name,
                    articles_fetched=len(raw_articles),
                    articles_inserted=inserted_count,
                    error=None
                ))
                logger.info(f"Collected {len(raw_articles)} articles from {source.name}, {inserted_count} new.")
                
            except MissingCredentialsError as e:
                logger.warning(f"Skipping {source.name}: {e}")
                summaries.append(SourceCollectionSummary(
                    source_name=source.name,
                    articles_fetched=0,
                    articles_inserted=0,
                    error=f"Skipped - {e}"
                ))
            except Exception as e:
                source_name = source.name
                logger.exception(f"Unexpected error collecting from {source_name}")
                # rollback in case of partial transaction failure for this source
                await db.rollback()
                summaries.append(SourceCollectionSummary(
                    source_name=source_name,
                    articles_fetched=0,
                    articles_inserted=0,
                    error=f"Failed - {sanitize_error(e, limit=200)}"
                ))
                
        logger.info("Collection cycle completed")
        
        # Trigger preprocessing immediately after collection
        # Note: If collection volume grows significantly, this synchronous invocation
        # should be moved to a background task queue (e.g. Celery) to avoid blocking.
        try:
            prep_orchestrator = PreprocessingOrchestrator()
            prep_summary = await prep_orchestrator.run_preprocessing_cycle(db)
            logger.info(f"Auto-preprocessing completed. Processed: {prep_summary.total_processed}")
        except Exception:
            logger.exception("Auto-preprocessing failed")
            
        return summaries

orchestrator = CollectionOrchestrator()
