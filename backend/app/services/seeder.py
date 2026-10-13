import logging

from sqlalchemy import select

from app.db.postgres import async_session_maker
from app.models.source import Source, SourceTypeEnum

logger = logging.getLogger(__name__)

INITIAL_SOURCES = [
    {
        "name": "BBC News (World)",
        "type": SourceTypeEnum.rss,
        "url_or_identifier": "https://feeds.bbci.co.uk/news/world/rss.xml",
    },
    {
        "name": "NewsAPI (Misinformation)",
        "type": SourceTypeEnum.news_api,
        "url_or_identifier": "misinformation",
    },
    {
        "name": "Reddit r/science",
        "type": SourceTypeEnum.reddit,
        "url_or_identifier": "science",
    }
]

async def seed_sources() -> None:
    try:
        async with async_session_maker() as db:
            result = await db.execute(select(Source).limit(1))
            existing = result.scalar_one_or_none()
            if existing is None:
                logger.info("No sources found. Seeding initial sources.")
                for s_data in INITIAL_SOURCES:
                    db.add(Source(**s_data))
                await db.commit()
                logger.info("Initial sources seeded successfully.")
    except Exception as e:
        logger.error(f"Failed to seed sources: {e}")
