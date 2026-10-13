"""Corpus-local checking and disclosed stored verdicts, including legacy scores."""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import Field as PField
from sqlalchemy import bindparam, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db_session
from app.core import ops_security
from app.db.postgres import async_session_maker
from app.models.claim import Claim
from app.models.evolution import FeedbackVerdict
from app.models.user import User
from app.services.verdict.engine import (
    DISPUTED_BANDS,
    METHOD_VERSION,
    SCORE_KIND,
    SUPPORTED_BANDS,
    NLIUnavailableError,
    Stance,
    engine_config,
)
from app.services.verdict.evidence import (
    Assessment,
    ClaimCheckResult,
    EmbeddingUnavailableError,
    PassageSource,
    check_claim,
)
from app.services.verdict.external import (
    ExternalAssessment,
    ExternalCheckResult,
    ExternalEvidenceUnavailableError,
    check_external_claim,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class VerdictCheckBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_text: str = PField(strict=True, min_length=10, max_length=2000)
    as_of: datetime | None = None
    limit: int = PField(default=6, strict=True, ge=1, le=12)

    @field_validator("claim_text", mode="before")
    @classmethod
    def trim_claim(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("as_of", mode="before")
    @classmethod
    def require_iso_datetime(cls, value: Any) -> Any:
        if (
            value is not None
            and not isinstance(value, datetime)
            and (
                not isinstance(value, str)
                or not re.match(r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}", value)
            )
        ):
            raise ValueError(
                "as_of must be an ISO datetime, not a date or epoch number"
            )
        return value


class VerdictCheckEvidence(BaseModel):
    claim_id: str
    article_id: str
    text: str
    passage: str
    passage_source: PassageSource
    url: str
    title: str
    domain: str | None
    published_at: datetime | None
    similarity: float | None
    stance: Stance
    syndication_group: str
    retrieval_method: Literal["vector", "keyword", "hybrid"] = "vector"
    collected_at: datetime | None = None
    extracted_at: datetime | None = None
    passage_start: int | None = None
    passage_end: int | None = None
    passage_sha256: str = ""
    abstention_reason: str | None = None


class VerdictCheckResponse(BaseModel):
    claim_text: str
    assessment: Assessment
    evidence: list[VerdictCheckEvidence]
    warnings: list[str]
    score_kind: Literal["uncalibrated_heuristic"]
    observed_at: datetime
    method_version: str


@router.post("/check", response_model=VerdictCheckResponse)
async def check_verdict(
    body: VerdictCheckBody,
    db: AsyncSession = Depends(get_db_session),
    _current_user: User = Depends(get_current_user),
) -> ClaimCheckResult:
    """Authenticated, read-only checking against the existing local corpus."""
    try:
        return await check_claim(
            db, body.claim_text, as_of=body.as_of, limit=body.limit
        )
    except (EmbeddingUnavailableError, NLIUnavailableError) as exc:
        logger.warning("Corpus checking model unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Corpus search is unavailable. Check PostgreSQL/pgvector and retry.",
        ) from exc


class ExternalCheckSource(BaseModel):
    title: str
    url: str
    passage: str
    stance: Stance
    abstention_reason: str | None
    revision_id: int
    published_at: datetime
    retrieved_at: datetime
    passage_start: int
    passage_end: int
    passage_sha256: str
    source_group: str


class ExternalCheckResponse(BaseModel):
    claim: str
    verdict: ExternalAssessment
    assessment: ExternalAssessment
    explanation: str
    sources: list[ExternalCheckSource]
    observed_at: datetime
    method_version: str
    warnings: list[str]


@router.post("/check-external", response_model=ExternalCheckResponse)
async def check_verdict_external(
    body: VerdictCheckBody,
    _current_user: User = Depends(get_current_user),
) -> ExternalCheckResult:
    """Revision-bound Wikipedia passages; local NLI requires no hosted API key."""
    try:
        return await check_external_claim(
            body.claim_text, as_of=body.as_of, limit=body.limit
        )
    except ExternalEvidenceUnavailableError as exc:
        logger.warning("External evidence unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _parse_evidence(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    return value if isinstance(value, dict) else None


def _score_metadata(value: Any) -> dict[str, Any]:
    evidence = _parse_evidence(value) or {}
    version = evidence.get("method_version")
    if version == METHOD_VERSION:
        warnings = evidence.get("warnings", [])
        return {
            "score_kind": SCORE_KIND,
            "method_version": version,
            "warnings": [w for w in warnings if isinstance(w, str)]
            if isinstance(warnings, list)
            else [],
            "score_disclosure": "The probability field is an uncalibrated heuristic, "
            "not a probability of truth.",
        }
    return {
        "score_kind": "legacy_unverified",
        "method_version": version if isinstance(version, str) else None,
        "warnings": [
            ("Legacy or unrecognized verdict evidence: stance-grounded support "
            "has not been verified; similarity-only support or self-derived "
            "outlet priors may have been used. This score is not a calibrated "
            "probability. Stored records have not been automatically rescored.")
        ],
    }


@router.get("")
async def list_verdicts(
    band: str = Query("", max_length=24),
    outlet: str = Query("", max_length=80),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    where = ["c.verdict IS NOT NULL"]
    params: dict[str, object] = {"limit": limit, "offset": offset}
    if band:
        where.append("c.verdict = :band")
        params["band"] = band
    if outlet:
        where.append("a.domain = :outlet")
        params["outlet"] = outlet
    where_sql = " AND ".join(where)
    try:
        # Count and page share one statement snapshot. The LEFT JOIN preserves
        # the true total even when offset is beyond the last matching item.
        rows = (
            (
                await db.execute(
                    text(f"""
            WITH matching AS (
                SELECT c.id, c.verdict_probability
                FROM claims c JOIN articles a ON a.id = c.article_id
                WHERE {where_sql}
            ), page AS (
                SELECT id, verdict_probability FROM matching
                ORDER BY verdict_probability ASC NULLS LAST, id ASC
                LIMIT :limit OFFSET :offset
            )
            SELECT totals.total, c.id::text AS id, c.claim_text, c.verdict,
                   c.verdict_probability AS probability,
                   c.confidence AS extraction_confidence,
                   a.domain AS outlet, a.title AS article_title, a.url AS article_url,
                   c.extracted_at, c.verdict_evidence AS score_evidence
            FROM (SELECT COUNT(*) AS total FROM matching) totals
            LEFT JOIN page ON TRUE
            LEFT JOIN claims c ON c.id = page.id
            LEFT JOIN articles a ON a.id = c.article_id
            ORDER BY page.verdict_probability ASC NULLS LAST, page.id ASC
        """),
                    params,
                )
            )
            .mappings()
            .all()
        )
    except SQLAlchemyError as exc:
        logger.exception("Stored verdict pagination failed")
        raise HTTPException(
            status_code=503, detail="Verdict storage is unavailable; retry later."
        ) from exc

    total = int(rows[0]["total"] or 0) if rows else 0
    items: list[dict[str, Any]] = []
    for row in rows:
        if row["id"] is None:
            continue
        item = dict(row)
        item.pop("total", None)
        item.update(_score_metadata(item.pop("score_evidence", None)))
        timestamp = item.get("extracted_at")
        item["extracted_at"] = (
            timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        )
        items.append(item)
    return {
        "total": total,
        "items": items,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(items) < total,
    }


@router.get("/stats")
async def verdict_stats() -> dict[str, Any]:
    async with async_session_maker() as db:
        dist = (
            await db.execute(
                text(
                    "SELECT verdict, COUNT(*), AVG(verdict_probability) "
                    "FROM claims GROUP BY verdict"
                )
            )
        ).all()
        total_claims = (await db.execute(text("SELECT COUNT(*) FROM claims"))).scalar()
        outlets = (
            await db.execute(
                text(
                    """
                    SELECT a.domain AS outlet,
                           COUNT(*) AS claims_total,
                           COUNT(*) FILTER (WHERE c.verdict IN :supported) AS supported,
                           COUNT(*) FILTER (WHERE c.verdict IN :disputed) AS disputed
                    FROM claims c JOIN articles a ON a.id = c.article_id
                    GROUP BY a.domain HAVING COUNT(*) > 0
                    ORDER BY claims_total DESC
                    """
                ).bindparams(
                    bindparam("supported", expanding=True),
                    bindparam("disputed", expanding=True),
                ),
                {
                    "supported": list(SUPPORTED_BANDS),
                    "disputed": list(DISPUTED_BANDS),
                },
            )
        ).all()

    def cred(sup: int, dis: int) -> float:
        # Compatibility display proxy from engine labels, NOT outlet reliability.
        alpha = 4.0
        return round((sup + alpha * 0.5) / (sup + dis + alpha), 3) if sup + dis else 0.5

    return {
        "total_claims": total_claims or 0,
        "warnings": [
            "Stored score aggregates may mix legacy and uncalibrated verdicts.",
            ("Outlet credibility is a compatibility proxy of engine labels, "
            "not independently measured outlet reliability."),
        ],
        "distribution": [
            {
                "band": r[0] or "UNSCORED",
                "count": r[1],
                "avg_probability": round(float(r[2]), 3) if r[2] is not None else None,
            }
            for r in dist
        ],
        "outlets": [
            {
                "name": r[0],
                "claims": r[1],
                "supported": r[2],
                "disputed": r[3],
                "credibility": cred(r[2], r[3]),
                "credibility_kind": "engine_label_proxy_not_outlet_reliability",
            }
            for r in outlets
        ],
    }


@router.get("/leaderboard")
async def outlet_leaderboard() -> dict[str, Any]:
    """Legacy engine-label proxy ranking, not independent outlet reliability."""
    stats = await verdict_stats()
    ranked = sorted(stats["outlets"], key=lambda o: (-o["credibility"], -o["claims"]))
    return {"ranking": ranked, "warnings": stats["warnings"]}


# NOTE: dynamic routes MUST stay below every static GET route in this file —
# "/{claim_id}" otherwise swallows "/summary", "/feedback" etc.


@router.get("/game/mutations")
async def mutation_chains(min_versions: int = 3, limit: int = 8) -> dict[str, Any]:
    """Claim mutation chains: same assertion, wording drifting outlet to
    outlet. Union-find over pgvector near-duplicate pairs (sim >= 0.90)."""
    import time as _time

    now = _time.time()
    cached = _MUT_CACHE.get("chains")
    if cached and now - cached[0] < _MUT_TTL:
        return {"chains": cached[1]}

    async with async_session_maker() as db:
        pairs = (
            await db.execute(
                text(
                    """
                    WITH chosen AS (
                        SELECT c.id, c.embedding, a.domain
                        FROM claims c
                        JOIN articles a ON a.id = c.article_id
                        WHERE c.embedding IS NOT NULL
                        ORDER BY c.extracted_at DESC
                        LIMIT 400
                    )
                    SELECT x.id::text AS src, y.id::text AS dst
                    FROM chosen x
                    JOIN LATERAL (
                        SELECT c2.id, a2.domain
                        FROM claims c2
                        JOIN articles a2 ON a2.id = c2.article_id
                        WHERE c2.embedding IS NOT NULL 
                          AND c2.id <> x.id
                          AND a2.domain <> x.domain
                        ORDER BY c2.embedding <=> x.embedding
                        LIMIT 5
                    ) y ON TRUE
                    WHERE (1 - (x.embedding <=> (SELECT embedding FROM claims WHERE id = y.id))) >= 0.84
                    LIMIT 4000
                    """
                )
            )
        ).all()

    parent: dict[str, str] = {}

    def find(a: str) -> str:
        while parent.get(a, a) != a:
            parent[a] = parent.get(parent[a], parent[a])
            a = parent[a]
        return a

    ids = set()
    edges = []
    for s_, d_ in pairs:
        ids.add(s_)
        ids.add(d_)
        edges.append((s_, d_))
    for i in ids:
        parent.setdefault(i, i)
    for s_, d_ in edges:
        ra, rb = find(s_), find(d_)
        if ra != rb:
            parent[ra] = rb

    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)

    big = [g for g in groups.values() if len(g) >= min_versions]
    big.sort(key=len, reverse=True)
    big = big[:limit]

    chains: list[dict[str, Any]] = []
    async with async_session_maker() as db:
        for g in big:
            rows = (
                await db.execute(
                    text(
                        """
                        SELECT c.id::text AS id, c.claim_text AS text,
                               a.domain AS domain, a.published_at AS pub,
                               a.title AS article_title
                        FROM claims c JOIN articles a ON a.id = c.article_id
                        WHERE c.id::text = ANY(:ids)
                        ORDER BY a.published_at ASC NULLS LAST
                        """
                    ),
                    {"ids": g},
                )
            ).all()
            versions = [
                {
                    "id": r[0],
                    "text": _v_clean(r[1], 220) or "",
                    "domain": _v_clean(r[2], 80),
                    "published_at": r[3].isoformat() if r[3] else None,
                    "article_title": _v_clean(r[4], 120),
                }
                for r in rows
            ]
            domains = {vv["domain"] for vv in versions}
            # a REAL mutation spans outlets; same-outlet boilerplate isn't one
            if len(versions) >= min_versions and len(domains) >= 2:
                chains.append(
                    {
                        "chain_id": g[0][:8],
                        "size": len(versions),
                        "distinct_outlets": len(domains),
                        "versions": versions,
                    }
                )
    chains.sort(key=lambda c: (-int(c["distinct_outlets"]), -int(c["size"])))
    result = chains[:limit]
    _MUT_CACHE["chains"] = (now, result)
    return {"chains": result}


def _v_clean(val: Any, limit: int) -> str | None:
    import html as h
    import re as re_

    if not val:
        return None
    s = h.unescape(re_.compile(r"<[^>]*>").sub(" ", str(val)))
    s = re_.sub(r"\s+", " ", s).strip()
    return s[:limit] or None


# ------------------------------------------------------- arcade


@router.get("/game/claim")
async def game_claim() -> dict[str, Any]:
    """Random verdicted claim for Fact-or-Fake. Answer NOT included —
    client reveals via /verdicts/{id} so peeking requires effort :)"""

    async with async_session_maker() as db:
        row = (
            await db.execute(
                text(
                    """
                    SELECT c.id::text AS id, c.claim_text AS text,
                           a.domain AS domain, a.published_at AS pub
                    FROM claims c JOIN articles a ON a.id = c.article_id
                    WHERE c.verdict IS NOT NULL
                      AND LENGTH(c.claim_text) BETWEEN 40 AND 240
                    ORDER BY random()
                    LIMIT 1
                    """
                )
            )
        ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="no verdicted claims yet")
    return {
        "id": row[0],
        "text": _v_clean(row[1], 260) or "",
        "domain": _v_clean(row[2], 80),
        "published_at": row[3].isoformat() if row[3] else None,
    }


# ------------------------------------------------- human feedback (HIL)


class VerdictFeedbackBody(BaseModel):
    claim_id: UUID
    vote: Literal["AGREE", "DISAGREE"]
    corrected_verdict: (
        Literal[
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "UNRESOLVED",
            "WEAKLY_CORROBORATED",
            "UNSUPPORTED",
            "DISPUTED",
        ]
        | None
    ) = None
    comment: str | None = PField(default=None, max_length=1000)


def _feedback_client_hash(request: Request) -> str:
    # Resolve dynamically so feedback and middleware share the canonical helper.
    # The private-name fallback only supports deployments awaiting its public alias.
    resolver = getattr(ops_security, "client_ip", None)
    if resolver is None:
        resolver = ops_security._client_ip
    return hashlib.sha256(resolver(request).encode("utf-8")).hexdigest()


@router.post("/feedback", status_code=201)
async def submit_verdict_feedback(
    body: VerdictFeedbackBody,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Anonymous, unverified feedback, not authoritative gold or calibration data.

    One vote per (claim, canonical client IP); re-voting atomically updates it.
    User-agent/header changes cannot create a different feedback identity.
    Shared IPs may share a vote; network identity is not a verified person.
    """
    try:
        claim_id = (
            await db.execute(select(Claim.id).where(Claim.id == body.claim_id))
        ).scalar_one_or_none()
        if claim_id is None:
            raise HTTPException(status_code=404, detail="claim not found")
        stmt = pg_insert(FeedbackVerdict).values(
            claim_id=claim_id,
            client_hash=_feedback_client_hash(request),
            vote=body.vote,
            corrected_verdict=body.corrected_verdict,
            comment=body.comment,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_verdict_feedback_claim_client",
            set_={
                "vote": stmt.excluded.vote,
                "corrected_verdict": stmt.excluded.corrected_verdict,
                "comment": stmt.excluded.comment,
                "created_at": func.now(),
            },
        )
        await db.execute(stmt)
        await db.commit()
    except SQLAlchemyError as exc:
        await db.rollback()
        logger.exception("Anonymous verdict feedback could not be stored")
        raise HTTPException(
            status_code=503, detail="Feedback storage is unavailable; retry later."
        ) from exc
    return {
        "status": "recorded",
        "claim_id": str(body.claim_id),
        "vote": body.vote,
        "feedback_kind": "anonymous_unverified",
        "is_ground_truth": False,
        "warnings": [
            ("Anonymous feedback is unverified, not authoritative gold "
            "or evidence of verdict accuracy. Shared IPs may share one vote.")
        ],
    }


@router.get("/summary")
async def verdict_summary() -> dict[str, Any]:
    """Disclosed config, stored-score distribution and unverified vote counts."""

    async with async_session_maker() as db:
        dist = (
            await db.execute(
                text(
                    "SELECT verdict, COUNT(*) AS n, AVG(verdict_probability) AS p "
                    "FROM claims WHERE verdict IS NOT NULL "
                    "GROUP BY verdict ORDER BY n DESC"
                )
            )
        ).all()
        total_scored = sum(r[1] for r in dist)

        fb = (
            await db.execute(
                text(
                    """
                    SELECT f.vote,
                           CASE WHEN f.corrected_verdict IS NOT NULL
                                THEN f.corrected_verdict ELSE c.verdict END
                                AS human_band,
                           c.verdict AS engine_band,
                           COUNT(*) AS n
                    FROM verdict_feedback f JOIN claims c ON c.id = f.claim_id
                    GROUP BY 1, 2, 3
                    """
                )
            )
        ).all()

    agrees = sum(r[3] for r in fb if r[0] == "AGREE")
    total_fb = sum(r[3] for r in fb)

    # A named alternate band is a reported disagreement, not a verified correction.
    # Keep the legacy metric key for clients, with an explicit non-gold disclosure.
    confirm_disputes = sum(
        r[3] for r in fb if r[0] == "DISAGREE" and r[2] and r[1] and r[1] != r[2]
    )

    return {
        "engine_config": engine_config(),
        "distribution": [
            {
                "band": r[0] or "UNSCORED",
                "count": r[1],
                "avg_probability": round(float(r[2]), 4) if r[2] is not None else None,
            }
            for r in dist
        ],
        "total_scored": total_scored,
        "warnings": [
            ("Stored aggregates may include legacy verdicts. Scores are "
            "uncalibrated heuristics, not probabilities of truth.")
        ],
        "human_feedback": {
            "feedback_kind": "anonymous_unverified",
            "is_ground_truth": False,
            "warning": "Anonymous agreement and named disagreements are unverified "
            "feedback, not authoritative gold, accuracy or calibration. "
            "confirmed_disagreements means a reported alternate band only.",
            "total_votes": total_fb,
            "agree": agrees,
            "disagree": total_fb - agrees,
            "agreement_rate": round(agrees / total_fb, 4) if total_fb else None,
            "confirmed_disagreements": confirm_disputes,
        },
    }


@router.get("/{claim_id}")
async def verdict_detail(
    claim_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Stored evidence and score provenance; reading never triggers rescoring."""
    try:
        row = (
            (
                await db.execute(
                    text("""
            SELECT c.id::text AS id, c.claim_text, c.verdict,
                   c.verdict_probability AS probability,
                   c.verdict_rationale, c.verdict_evidence,
                   c.confidence AS extraction_confidence,
                   a.domain AS outlet, a.title AS article_title,
                   a.url AS article_url, a.published_at,
                   COALESCE((SELECT json_agg(json_build_object('text', ce.entity_text, 'type', ce.entity_type))
                             FROM claim_entities ce WHERE ce.claim_id = c.id), '[]'::json) AS entities
            FROM claims c JOIN articles a ON a.id = c.article_id
            WHERE c.id = CAST(:cid AS uuid)
        """),
                    {"cid": str(claim_id)},
                )
            )
            .mappings()
            .first()
        )
    except SQLAlchemyError as exc:
        logger.exception("Stored verdict detail failed")
        raise HTTPException(
            status_code=503, detail="Verdict storage is unavailable; retry later."
        ) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="claim not found")
    item = dict(row)
    item["verdict_evidence"] = _parse_evidence(item.get("verdict_evidence"))
    item.update(_score_metadata(item["verdict_evidence"]))
    if isinstance(item.get("published_at"), datetime):
        item["published_at"] = item["published_at"].isoformat()
    return item


# ------------------------------------------------------- mutation chains

_MUT_CACHE: dict[str, tuple[float, object]] = {}
_MUT_TTL = 600.0  # seconds
