import json
import logging
import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any

from neo4j import AsyncManagedTransaction, AsyncSession
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession as PostgresSession
from sqlalchemy.orm import selectinload

from app.db.neo4j_client import neo4j_client
from app.models.article import Article
from app.models.claim import Claim
from app.models.evolution import ClaimRelationship, RelationshipTypeEnum
from app.models.source import Source

logger = logging.getLogger(__name__)

# Rows per UNWIND round-trip. 500 keeps each parameter payload well under
# Neo4j's bolt limits while cutting a 10k-claim sync from ~40k round-trips
# down to a couple of dozen statements.
_BATCH_SIZE = 500
# Includes retained corpora above 10k articles while keeping a finite bound.
MAX_PROJECTED_ARTICLES = 20000
MAX_PROJECTED_CLAIMS = 50000
MAX_PROJECTED_RELATIONSHIPS = 100000
GraphTransaction = AsyncManagedTransaction | AsyncSession

# The only relationship types allowed into the (unavoidably f-string'd)
# MERGE statement below. Values come from the StrEnum, so they are safe, but
# an explicit allow-list keeps that provably true under future edits.
_REL_TYPE_NAMES = frozenset(rt.value for rt in RelationshipTypeEnum)


def _chunks(rows: list[Any]) -> Iterator[list[Any]]:
    for i in range(0, len(rows), _BATCH_SIZE):
        yield rows[i : i + _BATCH_SIZE]


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class Neo4jWriter:
    """Idempotent Postgres -> Neo4j sync.

    Every write is a batched ``UNWIND`` — a handful of statements per cycle
    instead of 4+ round-trips per claim. After the merge, ``EVOLVED_FROM``
    edges rooted at claims *evaluated in this run* that the authoritative
    Postgres table no longer holds are deleted, so Neo4j mirrors the
    deduplicated relationship table instead of accumulating stale lineage
    from earlier, pre-dedup runs.
    """

    async def sync_persisted_graph(self, db: PostgresSession) -> dict[str, Any]:
        """Repair the graph from durable rows, without requiring NLP or clustering.

        Explicit caps prevent a maintenance action from loading an unbounded
        corpus. A cap failure leaves the existing graph untouched.
        """
        articles = list((await db.scalars(
            select(Article).order_by(Article.id).limit(MAX_PROJECTED_ARTICLES + 1)
        )).all())
        claims = list((await db.scalars(
            select(Claim).options(selectinload(Claim.entities))
            .order_by(Claim.id).limit(MAX_PROJECTED_CLAIMS + 1)
        )).all())
        relationships = list((await db.scalars(
            select(ClaimRelationship).order_by(ClaimRelationship.id)
            .limit(MAX_PROJECTED_RELATIONSHIPS + 1)
        )).all())
        if (len(articles) > MAX_PROJECTED_ARTICLES or len(claims) > MAX_PROJECTED_CLAIMS
                or len(relationships) > MAX_PROJECTED_RELATIONSHIPS):
            return {"success": False, "reason": "projection_capacity_exceeded",
                    "limits": {"articles": MAX_PROJECTED_ARTICLES,
                               "claims": MAX_PROJECTED_CLAIMS,
                               "relationships": MAX_PROJECTED_RELATIONSHIPS}}
        sources = (await db.scalars(
            select(Source).where(Source.id.in_({a.source_id for a in articles}))
        )).all()
        success = await self.sync_to_graph(
            claims, {a.id: a for a in articles}, {s.id: s for s in sources},
            relationships, evaluated_claim_ids={c.id for c in claims},
        )
        return {"success": success, "articles": len(articles), "claims": len(claims),
                "relationships": len(relationships),
                "reason": None if success else "graph_unavailable"}

    async def sync_to_graph(
        self,
        claims: Sequence[Claim],
        articles_by_id: dict[uuid.UUID, Article],
        sources_by_id: dict[uuid.UUID, Source],
        relationships: Sequence[ClaimRelationship],
        evaluated_claim_ids: set[uuid.UUID] | None = None,
    ) -> bool:
        """
        Synchronizes claims, entities, sources, and their relationships to Neo4j.
        This is an idempotent operation using MERGE.
        If it fails, it logs the error but does not raise, to avoid breaking Postgres transactions.

        ``evaluated_claim_ids`` are the claims whose lineage the mutation
        detector just re-examined; stale-edge pruning is scoped to them, so
        edges of claims *not* evaluated this run are never touched. Pruning
        is skipped entirely when the set is omitted.
        """
        try:
            driver = await neo4j_client.get_driver()
            async with driver.session() as session:
                await session.execute_write(
                    self._sync_transaction, claims, articles_by_id, sources_by_id,
                    relationships, evaluated_claim_ids,
                )
            logger.info(
                "Neo4j graph sync completed: %d claims, %d edges",
                len(claims),
                len(relationships),
            )
            return True
        except Exception:
            logger.exception("Failed to sync to Neo4j")
            return False

    async def _sync_transaction(
        self, tx: AsyncManagedTransaction, claims: Sequence[Claim],
        articles_by_id: dict[uuid.UUID, Article], sources_by_id: dict[uuid.UUID, Source],
        relationships: Sequence[ClaimRelationship],
        evaluated_claim_ids: set[uuid.UUID] | None,
    ) -> None:
        # One transaction prevents readers from seeing half-written projections.
        await self._merge_claims_entities_sources(tx, claims, articles_by_id, sources_by_id)
        await self._merge_relationships(tx, relationships)
        if evaluated_claim_ids:
            await self._prune_stale_evolved(tx, {str(cid) for cid in evaluated_claim_ids}, relationships)
        await self._project_article_links(tx, [str(aid) for aid in articles_by_id])

    async def _merge_claims_entities_sources(
        self,
        session: GraphTransaction,
        claims: Sequence[Claim],
        articles_by_id: dict[uuid.UUID, Article],
        sources_by_id: dict[uuid.UUID, Source],
    ) -> None:
        # 1. Sources — one UNWIND statement per batch
        source_rows = [
            {"id": str(s.id), "name": s.name, "type": s.type.value}
            for s in sources_by_id.values()
        ]
        for chunk in _chunks(source_rows):
            await session.run("""
                UNWIND $rows AS row
                MERGE (s:Source {id: row.id})
                SET s.name = row.name, s.type = row.type
            """, rows=chunk)

        # Articles are the actual graph/timeline UI contract. Publication time
        # is never substituted with collection or extraction time.
        article_rows = [
            {"id": str(a.id), "source_id": str(a.source_id), "title": a.title,
             "url": a.url, "domain": a.domain, "published_at": _isoformat(a.published_at),
             "content_hash": a.content_hash, "word_count": a.word_count,
             "canonical_id": str(a.canonical_article_id) if a.canonical_article_id else None}
            for a in articles_by_id.values()
        ]
        for chunk in _chunks(article_rows):
            await session.run("""
                UNWIND $rows AS row
                MERGE (a:Article {id: row.id})
                SET a.title = row.title, a.url = row.url, a.domain = row.domain,
                    a.publishedAt = row.published_at, a.contentHash = row.content_hash,
                    a.wordCount = row.word_count
                WITH a, row
                OPTIONAL MATCH (s:Source {id: row.source_id})
                FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
                    MERGE (a)-[:PUBLISHED_BY]->(s))
                FOREACH (_ IN CASE WHEN row.domain IS NULL OR row.domain = '' THEN [] ELSE [1] END |
                    MERGE (d:Domain {name: row.domain})
                    MERGE (a)-[:FROM_DOMAIN]->(d))
            """, rows=chunk)
        for chunk in _chunks(article_rows):
            await session.run("""
                UNWIND $rows AS row
                MATCH (a:Article {id: row.id})
                OPTIONAL MATCH (a)-[old:DUPLICATE_OF {projection: 'canonical_article'}]->()
                DELETE old
                WITH DISTINCT a, row
                MATCH (canonical:Article {id: row.canonical_id})
                WHERE a <> canonical
                MERGE (a)-[r:DUPLICATE_OF {projection: 'canonical_article'}]->(canonical)
                SET r.basis = 'persisted_canonical_article', r.observed_propagation = false
            """, rows=chunk)

        # Claims retain their article identity, including an explicit provenance link.
        claim_rows = [
            {
                "id": str(c.id), "text": c.claim_text,
                "article_id": str(c.article_id),
                "extracted_at": _isoformat(c.extracted_at),
                "published_at": _isoformat(articles_by_id[c.article_id].published_at)
                if c.article_id in articles_by_id else None,
            }
            for c in claims
        ]
        for chunk in _chunks(claim_rows):
            await session.run("""
                UNWIND $rows AS row
                MERGE (c:Claim {id: row.id})
                SET c.text = row.text, c.extracted_at = row.extracted_at,
                    c.article_id = row.article_id, c.published_at = row.published_at
                WITH c, row
                MATCH (a:Article {id: row.article_id})
                MERGE (c)-[:EXTRACTED_FROM]->(a)
            """, rows=chunk)

        # 3. Claim -> Source edges
        published_rows = [
            {
                "claim_id": str(c.id),
                "source_id": str(articles_by_id[c.article_id].source_id),
            }
            for c in claims
            if c.article_id in articles_by_id
        ]
        for chunk in _chunks(published_rows):
            await session.run("""
                UNWIND $rows AS row
                MATCH (c:Claim {id: row.claim_id})
                MATCH (s:Source {id: row.source_id})
                MERGE (c)-[:PUBLISHED_BY]->(s)
            """, rows=chunk)

        # 4. Entities and Claim -> Entity edges
        entity_rows = [
            {"claim_id": str(c.id), "text": e.entity_text, "type": e.entity_type}
            for c in claims
            for e in c.entities
        ]
        for chunk in _chunks(entity_rows):
            await session.run("""
                UNWIND $rows AS row
                MATCH (c:Claim {id: row.claim_id})
                MERGE (e:Entity {text: row.text, type: row.type})
                MERGE (c)-[:MENTIONS]->(e)
            """, rows=chunk)

    async def _merge_relationships(self, session: GraphTransaction, relationships: Sequence[ClaimRelationship]) -> None:
        by_type: dict[str, list[dict[str, Any]]] = {}
        for rel in relationships:
            rel_type = rel.relationship_type.value
            if rel_type not in _REL_TYPE_NAMES:
                logger.warning(
                    "Skipping unknown relationship type %r in graph sync", rel_type
                )
                continue
            by_type.setdefault(rel_type, []).append({
                "from_id": str(rel.from_claim_id),
                "to_id": str(rel.to_claim_id),
                "score": rel.score,
                "edge_id": str(rel.id) if rel.id else None,
                "created_at": rel.created_at.isoformat() if rel.created_at else None,
                # Neo4j properties cannot hold a nested JSON map. Retain the full
                # evidence as JSON plus queryable typed/provenance properties.
                "mutation_evidence": json.dumps(rel.mutation_evidence, sort_keys=True)
                if rel.mutation_evidence is not None else None,
                "mutation_types": (rel.mutation_evidence or {}).get("mutation_types", []),
                "algorithm_version": (rel.mutation_evidence or {}).get("algorithm_version"),
                "parent_claim_id": str(rel.to_claim_id)
                if rel.relationship_type == RelationshipTypeEnum.EVOLVED_FROM else None,
                "child_claim_id": str(rel.from_claim_id)
                if rel.relationship_type == RelationshipTypeEnum.EVOLVED_FROM else None,
                "observed_propagation": False,
            })
        for rel_type, rows in by_type.items():
            for chunk in _chunks(rows):
                # rel_type is allow-listed above, never user input.
                await session.run(f"""
                    UNWIND $rows AS row
                    MATCH (from_claim:Claim {{id: row.from_id}})
                    MATCH (to_claim:Claim {{id: row.to_id}})
                    MERGE (from_claim)-[r:{rel_type}]->(to_claim)
                    SET r.score = row.score, r.edge_id = row.edge_id,
                        r.created_at = row.created_at,
                        r.mutation_evidence = row.mutation_evidence,
                        r.mutation_types = row.mutation_types,
                        r.algorithm_version = row.algorithm_version,
                        r.parent_claim_id = row.parent_claim_id,
                        r.child_claim_id = row.child_claim_id,
                        r.inference = 'inferred',
                        r.observed_propagation = row.observed_propagation
                """, rows=chunk)

    async def _prune_stale_evolved(
        self,
        session: GraphTransaction,
        evaluated_claim_ids: set[str],
        relationships: Sequence[ClaimRelationship],
    ) -> None:
        """Delete EVOLVED_FROM edges the authoritative run no longer reproduces.

        Postgres prunes stale edges only for claims the mutation detector
        re-examined (``evaluated_claim_ids``); this mirrors that scoping so
        Neo4j cannot drift from the deduplicated relationship table. ``keep``
        is the current edge set for those same claims, filtered per batch so
        the parameter payload stays small.
        """
        keep_all = [
            (str(r.from_claim_id), str(r.to_claim_id))
            for r in relationships
            if r.relationship_type == RelationshipTypeEnum.EVOLVED_FROM
        ]
        for id_chunk in _chunks(sorted(evaluated_claim_ids)):
            chunk_set = set(id_chunk)
            keep = [pair for pair in keep_all if pair[0] in chunk_set]
            await session.run("""
                UNWIND $claim_ids AS cid
                MATCH (c:Claim {id: cid})-[r:EVOLVED_FROM]->(t:Claim)
                WHERE NOT [c.id, t.id] IN $keep
                DELETE r
            """, claim_ids=id_chunk, keep=[list(pair) for pair in keep])

    async def _project_article_links(self, session: GraphTransaction, article_ids: list[str]) -> None:
        """Aggregate real claim links for article views, retaining their inference basis.

        Only this projection's edges are reconciled. Independently built title
        similarity edges are never removed or relabeled as claim lineage.
        """
        for chunk in _chunks(article_ids):
            await session.run("""
                UNWIND $article_ids AS aid
                MATCH (a:Article {id: aid})-[r:SIMILAR {projection: 'claim_relationships'}]-()
                DELETE r
            """, article_ids=chunk)
        for chunk in _chunks(article_ids):
            await session.run("""
                UNWIND $article_ids AS aid
                MATCH (a:Article {id: aid})<-[:EXTRACTED_FROM]-(c:Claim)
                      -[r:SIMILAR_TO|EVOLVED_FROM]-(d:Claim)-[:EXTRACTED_FROM]->(b:Article)
                WHERE a.id < b.id
                WITH a, b, max(r.score) AS score,
                     collect(DISTINCT type(r)) AS relationship_types,
                     collect(DISTINCT r.edge_id) AS claim_edge_ids
                MERGE (a)-[link:SIMILAR {projection: 'claim_relationships'}]->(b)
                SET link.score = score, link.relationship_types = relationship_types,
                    link.claim_edge_ids = claim_edge_ids, link.inference = 'inferred_claim_relatedness',
                    link.observed_propagation = false
            """, article_ids=chunk)

