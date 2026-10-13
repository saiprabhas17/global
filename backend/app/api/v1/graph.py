"""Full-graph endpoint: every article node + every SIMILAR/DUPLICATE_OF edge.

The whole corpus graph in one payload (6.4k nodes, ~800 edges ≈ 1-2MB JSON).
"""

import asyncio
import html as htmllib
import logging
import re
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, HttpUrl, field_validator
from sqlalchemy import text

from app.api.deps import get_current_user
from app.db.neo4j_client import neo4j_client
from app.services.evolution.text_changes import analyze_text_change

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_SIMULATION_DEPTH = 6
MAX_SIMULATION_NODES = 2000
MAX_SIMULATION_EDGES = 6000


class MutationCompareRequest(BaseModel):
    older_text: str = Field(min_length=1, max_length=2000)
    newer_text: str = Field(min_length=1, max_length=2000)
    older_timestamp: datetime | None = None
    newer_timestamp: datetime | None = None
    context_note: str | None = Field(default=None, max_length=2000)
    provenance_url: HttpUrl | None = None
    include_summary: bool = False

    @field_validator("older_text", "newer_text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        # Preserve the input verbatim: analysis spans refer to these offsets.
        return value

class MutationSummaryRequest(BaseModel):
    article_ids: list[str] = Field(min_length=1, max_length=50)

@router.post("/timeline/mutation-summary", dependencies=[Depends(get_current_user)])
async def get_timeline_mutation_summary(payload: MutationSummaryRequest) -> dict[str, Any]:
    import uuid

    from sqlalchemy import select

    from app.db.postgres import async_session_maker
    from app.models.article import Article
    from app.models.claim import Claim
    from app.services.nlp.llm_client import get_llm_client
    
    # validate UUIDs
    valid_ids = []
    for aid in payload.article_ids:
        try:
            valid_ids.append(uuid.UUID(aid))
        except ValueError:
            pass
            
    if not valid_ids:
        return {"summary": "No valid article IDs provided."}
        
    async with async_session_maker() as db:
        claims_stmt = (
            select(Claim.claim_text)
            .join(Article, Claim.article_id == Article.id)
            .where(Claim.article_id.in_(valid_ids))
            .order_by(Article.published_at.asc().nulls_last(), Claim.id)
            .limit(50)
        )
        claims = list((await db.scalars(claims_stmt)).all())
        
    if len(claims) < 2:
        return {"summary": "Insufficient distinct claims in these articles to detect a temporal mutation."}
        
    llm = get_llm_client()
    summary = await llm.generate_mutation_summary(claims)
    return {"summary": summary}


@router.post("/mutation/compare", dependencies=[Depends(get_current_user)])
async def compare_mutations(payload: MutationCompareRequest) -> dict[str, Any]:
    from app.services.nlp.llm_client import get_llm_client
    analysis = await asyncio.to_thread(
        analyze_text_change, payload.older_text, payload.newer_text,
        older_timestamp=payload.older_timestamp,
        newer_timestamp=payload.newer_timestamp,
        context_note=payload.context_note,
        provenance_url=str(payload.provenance_url) if payload.provenance_url else None,
    )
    llm_summary = None
    summary_status = "not_requested"
    if payload.include_summary:
        try:
            llm = get_llm_client()
            llm_summary = await asyncio.wait_for(
                llm.generate_mutation_summary([payload.older_text, payload.newer_text]),
                timeout=30,
            )
            summary_status = "available"
        except Exception:
            logger.exception("Failed to generate optional mutation summary")
            summary_status = "unavailable"
        
    return {"analysis": analysis, "llm_summary": llm_summary, "summary_status": summary_status}


_TAG_RE = re.compile(r"<[^a-z/!]|</?[a-z][^>]*>", re.IGNORECASE)


def _clean_text(v: Any, limit: int = 200) -> str | None:
    """Strip any markup/entities from stored text before it reaches the UI."""
    if not v:
        return None
    s = htmllib.unescape(_TAG_RE.sub(" ", str(v)))
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:limit] or None)


@router.get("/full")
async def full_graph() -> dict[str, Any]:
    driver = await neo4j_client.get_driver()

    async def _run(tx: Any) -> Any:
        # nodes: all articles with their outlet + degree
        nodes_res = await tx.run(
            """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[e:SIMILAR]-()
            RETURN a.id AS id,
                   a.title AS title,
                   a.domain AS domain,
                   a.url AS url,
                   a.publishedAt AS published_at,
                   count(e) AS deg
            """
        )
        nodes = await nodes_res.data()

        edges_res = await tx.run(
            """
            MATCH (a:Article)-[e:SIMILAR]-(b:Article)
            RETURN a.id AS src, b.id AS dst, e.score AS score
            """
        )
        raw = await edges_res.data()

        dupe_res = await tx.run(
            """
            MATCH (a:Article)-[:DUPLICATE_OF]-(b:Article)
            RETURN a.id AS src, b.id AS dst
            """
        )
        dupes = await dupe_res.data()
        return nodes, raw, dupes

    async with driver.session() as ses:
        nodes, raw, dupes = await ses.execute_read(_run)

    seen = set()
    edges = []
    for e in raw:
        key = tuple(sorted((e["src"], e["dst"])))
        if key in seen:
            continue
        seen.add(key)
        edges.append({"src": e["src"], "dst": e["dst"], "score": round(float(e["score"]), 3)})
    for e in dupes:
        key = tuple(sorted((e["src"], e["dst"])))
        if key in seen:
            continue
        seen.add(key)
        edges.append({"src": e["src"], "dst": e["dst"], "score": 1.0})

    return {
        "nodes": [
            {
                "id": n["id"],
                "title": _clean_text(n.get("title"), 160) or "",
                "domain": _clean_text(n.get("domain"), 80),
                "url": n.get("url"),
                "published_at": n.get("published_at"),
                "deg": n.get("deg", 0),
            }
            for n in nodes
        ],
        "edges": edges,
        "counts": {"nodes": len(nodes), "edges": len(edges)},
    }


@router.get("/claims")
async def claims_graph(limit: int = 1200, threshold: float = 0.86) -> dict[str, Any]:
    """Claims network: persisted mutation edges, with a bounded pgvector fallback.

    Edges come from ``claim_relationships`` (EVOLVED_FROM / SIMILAR_TO rows the
    evolution pipeline already computed and indexed). Only when the selected
    claims have no persisted links do we fall back to per-row HNSW KNN via a
    LATERAL join. The previous implementation ran a full claims×claims cross
    join with a cosine-distance filter — ~20M distance computations per request
    (~120 s on the live corpus) while the UI polls this endpoint every 60 s.
    """
    from app.db.postgres import async_session_maker

    limit = max(50, min(int(limit), 3000))
    chosen_sql = """
        SELECT c.id::text AS id, c.claim_text AS title,
               a.domain AS domain,
               c.verdict AS verdict,
               c.verdict_probability AS prob,
               c.article_id::text AS article_id
        FROM claims c JOIN articles a ON a.id = c.article_id
        WHERE c.embedding IS NOT NULL AND c.verdict IS NOT NULL
        ORDER BY c.confidence DESC NULLS LAST, c.id
        LIMIT :lim
        """

    async with async_session_maker() as db:
        rows = (await db.execute(text(chosen_sql), {"lim": limit})).all()
        ids = {r[0] for r in rows}

        pair_rows = (
            await db.execute(
                text(
                    """
                    WITH chosen AS (
                        SELECT c.id
                        FROM claims c
                        WHERE c.embedding IS NOT NULL AND c.verdict IS NOT NULL
                        ORDER BY c.confidence DESC NULLS LAST, c.id
                        LIMIT :lim
                    )
                    SELECT r.from_claim_id::text AS src,
                           r.to_claim_id::text AS dst,
                           r.score AS sim,
                           r.relationship_type, r.mutation_evidence
                    FROM claim_relationships r
                    JOIN chosen f ON f.id = r.from_claim_id
                    JOIN chosen t ON t.id = r.to_claim_id
                    LIMIT 6000
                    """
                ),
                {"lim": limit},
            )
        ).all()
        edge_source = "claim_relationships"

        if not pair_rows:
            edge_source = "pgvector_knn"
            pair_rows = (
                await db.execute(
                    text(
                        """
                        WITH chosen AS (
                            SELECT c.id, c.embedding
                            FROM claims c
                            WHERE c.embedding IS NOT NULL AND c.verdict IS NOT NULL
                            ORDER BY c.confidence DESC NULLS LAST, c.id
                            LIMIT :lim
                        )
                        SELECT a.id::text AS src,
                               b.id::text AS dst,
                               1 - (a.embedding <=> b.embedding) AS sim,
                               'SIMILAR_TO' AS relationship_type,
                               NULL::jsonb AS mutation_evidence
                        FROM chosen a
                        JOIN LATERAL (
                            SELECT c2.id, c2.embedding
                            FROM claims c2
                            WHERE c2.embedding IS NOT NULL AND c2.id <> a.id
                            ORDER BY c2.embedding <=> a.embedding
                            LIMIT 4
                        ) b ON TRUE
                        WHERE 1 - (a.embedding <=> b.embedding) >= :thresh
                        LIMIT 6000
                        """
                    ),
                    {"lim": limit, "thresh": threshold},
                )
            ).all()

    nodes = [
        {
            "id": r[0], "title": _clean_text(r[1], 180) or "", "domain": _clean_text(r[2], 80),
            "verdict": r[3], "prob": float(r[4]) if r[4] is not None else None,
            "article_id": r[5],
        }
        for r in rows
    ]
    edges = []
    for r in pair_rows:
        if r[0] not in ids or r[1] not in ids:
            continue
        relationship_type = str(r[3])
        analysis = {**r[4], "observed_propagation": False} if isinstance(r[4], dict) else None
        edges.append({
            "src": r[0], "dst": r[1], "score": round(float(r[2] or 0.0), 3),
            "relationship_type": relationship_type,
            "parent_claim_id": r[1] if relationship_type == "EVOLVED_FROM" else None,
            "child_claim_id": r[0] if relationship_type == "EVOLVED_FROM" else None,
            "analysis": analysis, "observed_propagation": False,
            "inference": "inferred",
        })
    return {
        "nodes": nodes,
        "edges": edges,
        "counts": {"nodes": len(nodes), "edges": len(edges)},
        "edge_source": edge_source,
    }


async def _timeline_article_id(identifier: str) -> str:
    """Resolve article IDs and legacy claim-ID drill-downs without changing UI IDs."""
    from app.db.postgres import async_session_maker

    try:
        selected = str(uuid.UUID(identifier))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="article_id must be a UUID") from exc
    async with async_session_maker() as db:
        resolved = (
            await db.execute(text("""
                SELECT article_id FROM (
                    SELECT a.id::text AS article_id, 0 AS priority
                    FROM articles a WHERE a.id = CAST(:selected AS uuid)
                    UNION ALL
                    SELECT c.article_id::text AS article_id, 1 AS priority
                    FROM claims c WHERE c.id = CAST(:selected AS uuid)
                ) selected_articles
                ORDER BY priority LIMIT 1
            """), {"selected": selected})
        ).scalar_one_or_none()
    return str(resolved) if resolved is not None else selected


@router.get("/timeline")
async def timeline_clusters(
    limit: int = 60, article_id: str | None = None
) -> dict[str, Any]:
    """Story clusters ordered for the 3D timeline tunnel.

    Each ring = one hub story (most-linked article of its connected group),
    with member articles from other outlets. Rings ALWAYS run newest-first
    (index 0 = nearest camera): the tunnel's wheel/click dive moves to higher
    indices, so scrolling deeper reads back through the coverage in time —
    the last ring is the original first report of the story.

    When ``article_id`` is provided, only clusters within 3 hops of that
    article on SIMILAR edges are returned — used for the graph→timeline
    drill-down, so the tunnel shows the selected story alone.
    """
    limit = max(1, min(int(limit), 120))
    if article_id is not None:
        article_id = await _timeline_article_id(article_id)
    driver = await neo4j_client.get_driver()

    if article_id:
        # Focused mode: BFS from the selected article
        async def _run_focused(tx: Any) -> Any:
            res = await tx.run(
                """
                MATCH (root:Article {id: $aid})
                CALL {
                    WITH root
                    MATCH (root)-[:SIMILAR*1..3]-(a:Article)
                    RETURN DISTINCT a
                    UNION
                    WITH root
                    RETURN root AS a
                }
                WITH a
                MATCH (a)-[:SIMILAR]-(b:Article)
                WITH a, count(b) AS deg
                WHERE deg >= 1
                MATCH (a)-[s:SIMILAR]-(n:Article)
                WITH a, a.title AS title, a.domain AS domain, a.url AS url,
                     a.publishedAt AS published_at, deg,
                     collect({id: n.id, title: n.title, domain: n.domain,
                              url: n.url, publishedAt: n.publishedAt,
                              score: s.score})[0..12] AS members
                RETURN a.id AS id, title, domain, url, published_at, deg, members
                ORDER BY (published_at IS NULL) ASC, published_at DESC
                LIMIT $lim
                """,
                aid=article_id,
                lim=limit,
            )
            return await res.data()

        async with driver.session() as ses:
            rows = await ses.execute_read(_run_focused)
    else:
        # Default mode: all clusters newest-first
        async def _run(tx: Any) -> Any:
            res = await tx.run(
                """
                MATCH (a:Article)-[:SIMILAR]-(b:Article)
                WITH a, count(b) AS deg
                WHERE deg >= 1
                MATCH (a)-[s:SIMILAR]-(n:Article)
                WITH a, a.title AS title, a.domain AS domain, a.url AS url,
                     a.publishedAt AS published_at, deg,
                     collect({id: n.id, title: n.title, domain: n.domain,
                              url: n.url, publishedAt: n.publishedAt,
                              score: s.score})[0..8] AS members
                RETURN a.id AS id, title, domain, url, published_at, deg, members
                ORDER BY published_at DESC
                LIMIT $lim
                """,
                lim=limit,
            )
            return await res.data()

        async with driver.session() as ses:
            rows = await ses.execute_read(_run)

    clusters = []
    for r in rows:
        all_times = [r.get("published_at")] + [
            m.get("publishedAt") for m in (r.get("members") or [])
        ]
        times = sorted(
            [t for t in all_times if t], reverse=True
        )
        clusters.append(
            {
                "id": r["id"],
                "title": _clean_text(r.get("title"), 160) or "",
                "domain": _clean_text(r.get("domain"), 80),
                "url": r.get("url"),
                "published_at": str(r.get("published_at")) if r.get("published_at") else None,
                "newest_member_at": str(times[0]) if times else None,
                "deg": r.get("deg", 0),
                "members": [
                    {
                        "id": m.get("id"),
                        "title": _clean_text(m.get("title"), 140) or "",
                        "domain": _clean_text(m.get("domain"), 80),
                        "url": m.get("url"),
                        "published_at": str(m.get("publishedAt")) if m.get("publishedAt") else None,
                        "score": round(float(m.get("score") or 0), 3),
                    }
                    for m in (r.get("members") or [])
                ],
            }
        )

    return {
        "clusters": clusters,
        "count": len(clusters),
        "focused": article_id is not None,
        "article_id": article_id,
    }



@router.get("/scoops")
async def scoop_races(limit: int = 12) -> dict[str, Any]:
    """Who broke each story first? For every linked cluster: outlets ranked
    by publish time with exact lag behind the winner."""
    limit = max(1, min(int(limit), 50))
    driver = await neo4j_client.get_driver()

    async def _run(tx: Any) -> Any:
        res = await tx.run(
            """
            MATCH (a:Article)-[:SIMILAR]-(b:Article)
            WITH a, count(b) AS deg WHERE deg >= 2
            MATCH (m:Article)-[r:SIMILAR]-(a)
            WHERE m.publishedAt IS NOT NULL
            WITH a, deg, collect({id: m.id, title: m.title,
                 domain: m.domain, url: m.url,
                 publishedAt: m.publishedAt}) AS members
            RETURN a.id AS id, a.title AS title, a.domain AS domain,
                   a.url AS url, a.publishedAt AS hub_published,
                   deg, members
            ORDER BY hub_published DESC LIMIT $lim * 3
            """,
            lim=limit,
        )
        return await res.data()

    async with driver.session() as ses:
        rows = await ses.execute_read(_run)

    from datetime import datetime

    def ts(v: Any) -> datetime | None:
        if not v:
            return None
        if isinstance(v, datetime):
            return v
        try:
            return datetime.fromisoformat(str(v))
        except Exception:
            return None

    races = []
    for r in rows:
        entries = [
            {"id": r["id"], "title": _clean_text(r.get("title"), 140) or "",
             "domain": _clean_text(r.get("domain"), 80),
             "url": r.get("url"), "published": ts(r.get("hub_published"))}
        ]
        for m in r.get("members") or []:
            t = ts(m.get("publishedAt"))
            if t:
                entries.append({
                    "id": m["id"],
                    "title": _clean_text(m.get("title"), 140) or "",
                    "domain": _clean_text(m.get("domain"), 80),
                    "url": m.get("url"), "published": t,
                })
        # unique by id, drop undated
        seen = set()
        uniq = []
        for e in entries:
            if e["id"] in seen or e["published"] is None:
                continue
            seen.add(e["id"])
            uniq.append(e)
        if len(uniq) < 2:
            continue
        uniq.sort(key=lambda x: x["published"])
        winner = uniq[0]
        lag_total = (uniq[-1]["published"] - winner["published"]).total_seconds()
        racers = []
        for i, e in enumerate(uniq[:6]):
            lag_s = (e["published"] - winner["published"]).total_seconds()
            racers.append({
                **e,
                "published": e["published"].isoformat(),
                "lag_seconds": int(lag_s),
                "position": i,
            })
        races.append({
            "story": _clean_text(winner["title"], 120) or "",
            "winner_domain": winner["domain"],
            "racers": racers,
            "field_size": len(uniq),
            "lag_spread_seconds": int(lag_total),
        })
        if len(races) >= limit:
            break

    return {"races": races, "count": len(races)}


@router.get("/simulate")
async def simulate_spread(hub: str, p: float = 0.5, max_depth: int = 3) -> dict[str, Any]:
    """Hypothetical branching simulation on a bounded hub neighborhood.

    Read one-hop frontiers rather than the entire graph or an unbounded
    variable-length path expansion. SIMILAR is not observed transmission.
    """
    from app.services.simulate import simulate_cascade

    p = max(0.05, min(0.95, p))
    max_depth = max(1, min(int(max_depth), MAX_SIMULATION_DEPTH))
    driver = await neo4j_client.get_driver()

    async def _run(tx: Any) -> Any:
        hub_res = await tx.run(
            "MATCH (a:Article {id: $hub}) RETURN a.title AS title, a.domain AS domain",
            hub=hub,
        )
        hub_rows = await hub_res.data()
        if not hub_rows:
            return [], None, False
        frontier, seen_nodes = [hub], {hub}
        unique_edges: set[tuple[str, str]] = set()
        truncated = False
        for _ in range(max_depth):
            if not frontier:
                break
            result = await tx.run("""
                UNWIND $frontier AS cid
                MATCH (a:Article {id: cid})-[:SIMILAR]-(b:Article)
                RETURN DISTINCT a.id AS src, b.id AS dst
                ORDER BY src, dst LIMIT $lim
            """, frontier=frontier, lim=MAX_SIMULATION_EDGES + 1)
            rows = await result.data()
            if len(rows) > MAX_SIMULATION_EDGES:
                truncated = True
            frontier = []
            for row in rows[:MAX_SIMULATION_EDGES]:
                a, b = str(row["src"]), str(row["dst"])
                key = (min(a, b), max(a, b))
                if key in unique_edges or key[0] == key[1]:
                    continue
                new_ids = set(key) - seen_nodes
                if len(seen_nodes) + len(new_ids) > MAX_SIMULATION_NODES:
                    truncated = True
                    continue
                if len(unique_edges) == MAX_SIMULATION_EDGES:
                    truncated = True
                    break
                unique_edges.add(key)
                for cid in sorted(new_ids):
                    seen_nodes.add(cid)
                    frontier.append(cid)
            if len(unique_edges) == MAX_SIMULATION_EDGES or len(seen_nodes) == MAX_SIMULATION_NODES:
                truncated = True
                break
        return [{"src": a, "dst": b} for a, b in sorted(unique_edges)], hub_rows[0], truncated

    async with driver.session() as ses:
        edges, hub_node, truncated = await ses.execute_read(_run)
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge["src"], []).append(edge["dst"])
        adjacency.setdefault(edge["dst"], []).append(edge["src"])
    if hub not in adjacency:
        raise HTTPException(status_code=404, detail="hub article not in the link graph")
    return {
        "hub": {"id": hub, "title": _clean_text(hub_node.get("title"), 140),
                "domain": _clean_text(hub_node.get("domain"), 80)},
        **simulate_cascade(adjacency, hub, p=p, max_depth=max_depth),
        "graph_scope": "bounded_hub_neighborhood",
        "truncated": truncated, "observed_propagation": False,
        "limits": {"nodes": MAX_SIMULATION_NODES, "edges": MAX_SIMULATION_EDGES,
                   "depth": MAX_SIMULATION_DEPTH},
    }
