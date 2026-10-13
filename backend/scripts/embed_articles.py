"""Backfill article vectors using the deployed model and the migrated schema.

Run `alembic upgrade head` first. This script never creates schema or changes
an existing embedding. Changing EMBEDDING_MODEL requires an explicit migration
of *both* article and claim vectors; mixing vector spaces is unsupported.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, update

from app.db.postgres import async_session_maker
from app.models.article import Article, ProcessingStatusEnum
from app.services.nlp.embedding_service import EmbeddingService


async def backfill(limit: int = 100) -> int:
    if limit < 1:
        raise ValueError("Batch limit must be positive")
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(Article.id, Article.title, Article.cleaned_content, Article.content)
            .where(Article.embedding.is_(None),
                   Article.processing_status == ProcessingStatusEnum.processed)
            .order_by(Article.id).limit(limit)
        )).all()
    for aid, title, cleaned, content in rows:
        vector = await asyncio.to_thread(
            EmbeddingService.generate_document_embedding,
            f"{title}. {cleaned or content or ''}", allow_load=True,
        )
        # Commit each finished article so an interrupted backfill resumes safely.
        async with async_session_maker() as db:
            await db.execute(update(Article).where(
                Article.id == aid, Article.embedding.is_(None),
            ).values(embedding=vector))
            await db.commit()
    return len(rows)


async def main() -> None:
    total = 0
    while (count := await backfill()):
        total += count
        print(f"backfilled {total} embeddings", flush=True)
    print(f"embedding backfill complete: {total} articles")


if __name__ == "__main__":
    asyncio.run(main())
