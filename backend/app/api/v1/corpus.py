"""Read-only endpoints exposing the real corpus + story graph."""

import asyncio
import html as htmllib
import logging
import re
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session
from app.db.neo4j_client import neo4j_client
from app.models.article import Article, ProcessingStatusEnum
from app.models.source import Source
from app.services.nlp.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------- articles


def _article_row(article: Article, *, body: bool = False) -> dict[str, Any]:
    result = {
        "id": str(article.id), "title": _clean_text(article.title, 500),
        "url": article.url, "author": _clean_text(article.author, 300),
        "domain": article.domain, "language": article.language,
        "published_at": article.published_at, "word_count": article.word_count,
        "nlp_status": article.nlp_status.value,
    }
    content = article.cleaned_content or article.content or ""
    if body:
        result.update(body=content, collected_at=article.collected_at)
    else:
        result["excerpt"] = _clean_text(content)
    return result


@router.get("/articles")
async def list_articles(
    q: str = Query("", max_length=120),
    domain: str = Query("", max_length=80),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Paginated live corpus listing; user search treats %, _ as literal text."""
    filters = [Article.processing_status == ProcessingStatusEnum.processed]
    if q.strip():
        filters.append(or_(
            Article.title.icontains(q.strip(), autoescape=True),
            Article.author.icontains(q.strip(), autoescape=True),
        ))
    if domain:
        filters.append(Article.domain == domain)
    total = await db.scalar(select(func.count(Article.id)).where(*filters))
    articles = (await db.scalars(
        select(Article).where(*filters)
        .order_by(Article.published_at.desc().nullslast(), Article.id)
        .limit(limit).offset(offset)
    )).all()
    domains = (await db.execute(
        select(Article.domain, func.count(Article.id).label("count"))
        .where(Article.processing_status == ProcessingStatusEnum.processed,
               Article.domain.is_not(None))
        .group_by(Article.domain).order_by(func.count(Article.id).desc(), Article.domain)
    )).all()
    return {
        "total": total or 0, "limit": limit, "offset": offset,
        "items": [_article_row(article) for article in articles],
        "domains": [{"name": domain, "count": count} for domain, count in domains],
    }


@router.get("/articles/recent")
async def recent_articles(
    limit: int = Query(12, ge=1, le=40),
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Most recently collected processed articles from the live database."""
    rows = (await db.execute(
        select(Article, Source.name)
        .outerjoin(Source, Source.id == Article.source_id)
        .where(Article.processing_status == ProcessingStatusEnum.processed)
        .order_by(Article.collected_at.desc(), Article.id).limit(limit)
    )).all()
    return {"items": [
        {**_article_row(article), "collected_at": article.collected_at, "source_name": name}
        for article, name in rows
    ]}


@router.get("/stats")
async def corpus_stats(db: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Corpus statistics work immediately after migrations, including an empty DB."""
    row = (await db.execute(
        select(func.count(Article.id), func.count(Article.embedding),
               func.count(func.distinct(Article.domain)),
               func.min(Article.published_at), func.max(Article.published_at))
        .where(Article.processing_status == ProcessingStatusEnum.processed)
    )).one()
    nlp = (await db.execute(
        select(Article.nlp_status, func.count(Article.id)).group_by(Article.nlp_status)
    )).all()
    return {
        "total": row[0], "embedded": row[1], "domains": row[2],
        "earliest": row[3].date().isoformat() if row[3] else None,
        "latest": row[4].date().isoformat() if row[4] else None,
        "nlp_counts": {status.value: count for status, count in nlp},
    }


@router.get("/search")
async def semantic_search(
    q: str = Query(..., min_length=2, max_length=300),
    limit: int = Query(10, ge=1, le=30),
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Nearest-neighbor article search; first-use loading stays off the event loop."""
    query = q.strip()
    if len(query) < 2:
        raise HTTPException(status_code=422, detail="Search needs at least two non-space characters.")
    indexed = await db.scalar(select(func.count(Article.id)).where(
        Article.embedding.is_not(None),
        Article.processing_status == ProcessingStatusEnum.processed,
    ))
    if not indexed:
        return {"query": query, "count": 0, "items": []}
    try:
        qvec = await asyncio.to_thread(
            EmbeddingService.generate_embedding, query, allow_load=True,
        )
    except Exception as exc:
        logger.exception("Article search embedding unavailable")
        raise HTTPException(
            status_code=503,
            detail="Embedding model unavailable. Check model installation or provider access.",
        ) from exc
    distance = Article.embedding.cosine_distance(qvec)
    rows = (await db.execute(
        select(Article, (1 - distance).label("score"))
        .where(Article.embedding.is_not(None),
               Article.processing_status == ProcessingStatusEnum.processed)
        .order_by(distance, Article.id).limit(limit)
    )).all()
    return {
        "query": query, "count": len(rows),
        "items": [{**_article_row(article), "score": round(float(score), 4)}
                  for article, score in rows],
    }


@router.get("/articles/{article_id}")
async def article_detail(
    article_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    article = await db.get(Article, article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="article not found")
    return _article_row(article, body=True)


# ---------------------------------------------------------------- graph


@router.get("/graph/story-clusters")
async def story_clusters(min_links: int = Query(3, ge=1), limit: int = Query(6, ge=1, le=20)) -> dict[str, Any]:
    """Hub stories (most SIMILAR links) with their cross-outlet neighbors."""
    driver = await neo4j_client.get_driver()

    async def _run(tx: Any) -> Any:
        hubs_res = await tx.run(
            """
            MATCH (a:Article)-[e:SIMILAR]-(b:Article)
            WITH a, count(e) AS deg, sum(e.score) AS mass
            WHERE deg >= $min_links
            RETURN a.id AS id, a.title AS title, a.domain AS domain,
                   a.url AS url, deg, mass
            ORDER BY deg DESC LIMIT $limit
            """,
            min_links=min_links,
            limit=limit,
        )
        hubs = await hubs_res.data()
        out = []
        for h in hubs:
            nbrs = await tx.run(
                """
                MATCH (a:Article {id: $id})-[e:SIMILAR]-(b:Article)
                RETURN b.id AS id, b.title AS title, b.domain AS domain,
                       b.url AS url, e.score AS score
                ORDER BY e.score DESC
                """,
                id=h["id"],
            )
            nbrs_data = await nbrs.data()
            out.append({**h, "neighbors": nbrs_data})
        return out

    async with driver.session() as ses:
        clusters = await ses.execute_read(_run)
    return {"clusters": clusters}


@router.get("/graph/stats")
async def graph_stats() -> dict[str, Any]:
    driver = await neo4j_client.get_driver()

    async def _run(tx: Any) -> Any:
        stats = {}
        for label, q in [
            ("articles", "MATCH (a:Article) RETURN count(a)"),
            ("domains", "MATCH (d:Domain) RETURN count(d)"),
            ("similar", "MATCH ()-[e:SIMILAR]->() RETURN count(e)"),
            ("dupes", "MATCH ()-[e:DUPLICATE_OF]->() RETURN count(e)"),
        ]:
            res = await tx.run(q)
            stats[label] = (await res.single())[0]

        outlets_res = await tx.run(
            """
            MATCH (a:Article)-[:FROM_DOMAIN]->(d:Domain)
            OPTIONAL MATCH (a)-[e:SIMILAR]-(b:Article)
            WHERE b.domain IS NOT NULL AND b.domain <> d.name
            RETURN d.name AS name, count(DISTINCT a) AS stories,
                   count(DISTINCT CASE WHEN b IS NOT NULL THEN b END) AS cross
            ORDER BY stories DESC
            """)
        stats["outlets"] = await outlets_res.data()
        return stats

    async with driver.session() as ses:
        return await ses.execute_read(_run)


_TAG_RE = re.compile(r"<[^a-z/!]|</?[a-z][^>]*>", re.IGNORECASE)


def _clean_text(v: Any, limit: int = 240) -> str | None:
    """Strip any markup/entities from stored text before it reaches the UI."""
    if not v:
        return None
    s2 = htmllib.unescape(_TAG_RE.sub(" ", str(v)))
    s2 = re.sub(r"\s+", " ", s2).strip()
    return (s2[:limit] or None)
