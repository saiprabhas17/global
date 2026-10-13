import logging
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.models.article import Article
from app.models.claim import Claim
from app.models.evolution import ClaimClusterAssignment, ClaimClusterRun
from app.models.source import Source
from app.services.evolution.cluster_service import (
    CLUSTERING_ALGORITHM_VERSION,
    ClusterService,
)
from app.services.evolution.mutation_detector import (
    MUTATION_ALGORITHM_VERSION,
    MutationDetector,
    candidate_limits,
)
from app.services.evolution.neo4j_writer import Neo4jWriter
from app.services.evolution.text_changes import (
    ALGORITHM_VERSION as TEXT_CHANGE_ALGORITHM_VERSION,
)

logger = logging.getLogger(__name__)


class EvolutionOrchestrator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.cluster_service = ClusterService()
        self.mutation_detector = MutationDetector()
        self.neo4j_writer = Neo4jWriter()

    async def run_evolution_cycle(self, db: AsyncSession, force: bool = False) -> dict[str, Any]:
        """
        Executes the evolution engine: clustering, mutation detection, graph sync.
        Respects guards for minimum corpus size and debounce (new claims).
        """
        # 1. Corpus size check
        total_claims = await db.scalar(
                    select(func.count(Claim.id)).where(Claim.embedding.is_not(None))
                ) or 0
        min_corpus = self.settings.MIN_CORPUS_SIZE_FOR_CLUSTERING

        if total_claims < min_corpus:
            logger.info(f"Skipping evolution cycle: corpus size {total_claims} < {min_corpus}.")
            return {
                "status": "insufficient_data",
                "claims_available": total_claims,
                "minimum_required": min_corpus,
                "graph_projection": await self.neo4j_writer.sync_persisted_graph(db),
            }

        # 2. Debounce guard
        if not force:
            last_run = await db.scalar(
                select(ClaimClusterRun).order_by(ClaimClusterRun.run_at.desc()).limit(1)
            )
            if last_run:
                new_claims = total_claims - last_run.claims_in_corpus
                min_new = self.settings.MIN_NEW_CLAIMS_TO_RECLUSTER
                # Historical runs counted null embeddings and used different
                # parent semantics. Re-evaluate once on algorithm upgrade,
                # rather than letting their corpus count debounce indefinitely.
                current_algorithm = (last_run.algorithm_params or {}).get("algorithm_version") == CLUSTERING_ALGORITHM_VERSION
                if current_algorithm and new_claims < min_new:
                    logger.info(f"Skipping evolution cycle: {new_claims} new claims < {min_new}.")
                    return {
                        "status": "debounced",
                        "new_claims": new_claims,
                        "minimum_new_required": min_new,
                        "last_run_at": last_run.run_at.isoformat(),
                        "graph_projection": await self.neo4j_writer.sync_persisted_graph(db),
                    }

        logger.info(f"Starting evolution cycle (force={force}). Corpus size: {total_claims}")

        # Fetch all claims with their entities
        stmt = select(Claim).options(selectinload(Claim.entities))
        res = await db.execute(stmt)
        claims: Sequence[Claim] = res.scalars().all()

        # Build quick lookups for articles and sources
        article_ids = {c.article_id for c in claims}
        art_res = await db.execute(select(Article).where(Article.id.in_(article_ids)))
        articles_by_id = {a.id: a for a in art_res.scalars().all()}

        source_ids = {a.source_id for a in articles_by_id.values()}
        src_res = await db.execute(select(Source).where(Source.id.in_(source_ids)))
        sources_by_id = {s.id: s for s in src_res.scalars().all()}

        claims_by_id = {c.id: c for c in claims}

        # 3. Clustering
        try:
            cluster_run = await self.cluster_service.run_clustering(db, claims)
        except ImportError as e:
            logger.error(f"Clustering dependencies missing: {e}")
            return {"status": "error", "error": "BERTopic dependencies missing"}

        # 4. Mutation Detection
        # Fetch the assignments just created in the current transaction with an
        # explicit awaited SELECT: touching the lazy `cluster_run.assignments`
        # collection here would emit IO outside the async greenlet context and
        # raise sqlalchemy.exc.MissingGreenlet.
        assignments = (
            await db.execute(
                select(ClaimClusterAssignment).where(
                    ClaimClusterAssignment.cluster_run_id == cluster_run.id
                )
            )
        ).scalars().all()
        relationships = await self.mutation_detector.run_mutation_detection(
            db, assignments, claims_by_id, articles_by_id
        )

        candidate_limit, max_lag_days = candidate_limits(self.settings)
        cluster_run.algorithm_params = {
            **(cluster_run.algorithm_params or {}),
            "mutation_algorithm_version": MUTATION_ALGORITHM_VERSION,
            "text_change_algorithm_version": TEXT_CHANGE_ALGORITHM_VERSION,
            "mutation_candidate_limit": candidate_limit,
            "mutation_max_lag_days": max_lag_days,
            "evolution_similarity_threshold": self.settings.EVOLUTION_SIMILARITY_THRESHOLD,
            "similar_to_threshold": self.settings.SIMILAR_TO_THRESHOLD,
        }

        # Commit Postgres transaction so we have durable IDs before Neo4j
        await db.commit()
        await db.refresh(cluster_run)

        # 5. Neo4j Sync
        neo4j_success = await self.neo4j_writer.sync_to_graph(
            claims, articles_by_id, sources_by_id, relationships,
            # All supplied children were reconciled, including noise, singleton
            # topics and claims whose embeddings disappeared between runs.
            evaluated_claim_ids=set(claims_by_id),
        )

        valid_clusters = len({a.topic_id for a in assignments if a.topic_id != -1})

        return {
            "status": "success",
            "corpus_size": total_claims,
            "clusters_found": valid_clusters,
            "relationships_detected": len(relationships),
            "neo4j_sync_success": neo4j_success
        }
