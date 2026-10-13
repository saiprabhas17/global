"""Bounded, corpus-local claim checking with traceable passages and no truth odds.

No URL is fetched and no remote inference API is used. The existing claim
embedding space supplies retrieval only. Stance is measured on the returned
passage, not inferred from cosine, article style, or outlet reputation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, NotRequired, TypedDict

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.verdict.engine import (
    METHOD_VERSION,
    SCORE_KIND,
    Stance,
    canonical_domain,
    classify_stance,
    near_window,
)

logger = logging.getLogger(__name__)
EMBEDDING_DIMENSIONS = 768  # Existing claims.embedding schema/model space.
MAX_CANDIDATES = 96
MAX_CONTENT_CHARS = 32_000
MAX_PASSAGE_CHARS = 1_000
MAX_EVIDENCE = 12

Assessment = Literal[
    "SUPPORTED_BY_CORPUS",
    "CONTRADICTED_BY_CORPUS",
    "MIXED_EVIDENCE",
    "INSUFFICIENT_EVIDENCE",
]
PassageSource = Literal["cleaned_content", "content", "extracted_claim"]

CORPUS_WARNING = (
    "Corpus-local coverage only: bounded keyword/vector retrieval over stored claims, "
    "not a web search or independent fact verification. Missing evidence is not falsity."
)
INDEPENDENCE_WARNING = (
    "Canonical/content duplicates are grouped; different domains still do not "
    "guarantee independent reporting. NLI measures textual stance, not truth."
)


class EmbeddingUnavailableError(RuntimeError):
    """The query cannot be embedded compatibly with the stored corpus."""


class EvidenceItem(TypedDict):
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
    retrieval_method: NotRequired[Literal["vector", "keyword", "hybrid"]]
    collected_at: NotRequired[datetime | None]
    extracted_at: NotRequired[datetime | None]
    passage_start: NotRequired[int | None]
    passage_end: NotRequired[int | None]
    passage_sha256: NotRequired[str]
    abstention_reason: NotRequired[str | None]


class ClaimCheckResult(TypedDict):
    claim_text: str
    assessment: Assessment
    evidence: list[EvidenceItem]
    warnings: list[str]
    score_kind: Literal["uncalibrated_heuristic"]
    observed_at: datetime
    method_version: str


@dataclass(frozen=True)
class EvidenceCandidate:
    claim_id: str
    article_id: str
    text: str
    url: str
    title: str
    domain: str | None
    published_at: datetime | None
    similarity: float | None
    collected_at: datetime | None = None
    extracted_at: datetime | None = None
    canonical_article_id: str | None = None
    content_hash: str | None = None
    cleaned_content: str | None = None
    content: str | None = None
    syndication_group: str = ""
    lexical_score: float = 0.0
    retrieval_score: float | None = None
    retrieval_method: Literal["vector", "keyword", "hybrid"] = "vector"


def as_utc(value: datetime) -> datetime:
    """Naive corpus timestamps/ISO datetimes are explicitly interpreted as UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise TypeError("Invalid corpus timestamp")
    return as_utc(value)


def vector_literal(embedding: Sequence[float] | str) -> str:
    """Bind pgvector as a literal with a cast, never interpolate a vector into SQL."""
    try:
        raw = json.loads(embedding) if isinstance(embedding, str) else embedding
        values = [float(value) for value in raw]
        if (
            len(values) != EMBEDDING_DIMENSIONS
            or not all(math.isfinite(value) for value in values)
            or not any(value != 0 for value in values)
        ):
            raise ValueError("Invalid embedding dimension, nonfinite or zero vector")
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingUnavailableError(
            "A finite, nonzero 768-dimensional embedding compatible with the "
            "existing claims corpus is required. Check EMBEDDING_MODEL and the "
            "stored embedding space; do not mix models without re-embedding."
        ) from exc
    return json.dumps(values, separators=(",", ":"), allow_nan=False)


async def embed_query(claim_text: str) -> list[float]:
    """Use the existing loaded embedding service; do not download on request."""
    try:
        from app.services.nlp.embedding_service import EmbeddingService

        embedding = await asyncio.to_thread(
            EmbeddingService.generate_embedding, claim_text
        )
        vector_literal(embedding)
        return embedding
    except EmbeddingUnavailableError:
        raise
    except Exception as exc:
        raise EmbeddingUnavailableError(
            "The local corpus embedding model is unavailable. Load the configured "
            "EMBEDDING_MODEL with EmbeddingService at startup and pre-cache its "
            "weights/dependencies. It must match the existing 768-dimensional "
            "claims embeddings. Claim checking does not download models on request."
        ) from exc


def _candidate(row: Mapping[str, Any]) -> EvidenceCandidate:
    similarity = float(row["similarity"]) if row.get("similarity") is not None else None
    if similarity is not None and (
        not math.isfinite(similarity) or not -1.001 <= similarity <= 1.001
    ):
        raise ValueError("Invalid corpus similarity")
    lexical_score = float(row.get("lexical_score", 0.0))
    retrieval_score = row.get("retrieval_score")
    if not math.isfinite(lexical_score) or lexical_score < 0:
        raise ValueError("Invalid lexical retrieval score")
    if retrieval_score is not None:
        retrieval_score = float(retrieval_score)
        if not math.isfinite(retrieval_score) or retrieval_score < 0:
            raise ValueError("Invalid fusion retrieval score")
    method: Literal["vector", "keyword", "hybrid"] = "vector"
    if lexical_score > 0:
        method = "hybrid" if similarity is not None else "keyword"
    return EvidenceCandidate(
        claim_id=str(row["claim_id"]),
        article_id=str(row["article_id"]),
        text=str(row["text"]),
        url=str(row.get("url") or ""),
        title=str(row.get("title") or ""),
        domain=canonical_domain(row.get("domain")),
        published_at=_timestamp(row.get("published_at")),
        collected_at=_timestamp(row.get("collected_at")),
        extracted_at=_timestamp(row.get("extracted_at")),
        similarity=max(-1.0, min(1.0, similarity)) if similarity is not None else None,
        lexical_score=lexical_score,
        retrieval_score=retrieval_score,
        retrieval_method=method,
        canonical_article_id=(
            str(row["canonical_article_id"])
            if row.get("canonical_article_id")
            else None
        ),
        content_hash=row.get("content_hash"),
        cleaned_content=row.get("cleaned_content"),
        content=row.get("content"),
    )


async def retrieve_candidates(
    db: AsyncSession,
    embedding: Sequence[float] | str | None,
    *,
    cutoff: datetime,
    historical: bool,
    limit: int,
    exclude_claim_id: str | None = None,
    query_text: str = "",
) -> tuple[list[EvidenceCandidate], list[str]]:
    """Bound each retrieval channel, then fuse ranks, preserving temporal filters.

    Rank fusion is a retrieval ordering, never a probability or stance. Keyword
    retrieval includes claims whose embeddings are absent. Both channels use the
    same publication/collection/extraction cutoff before their LIMIT clauses.
    """
    candidate_limit = min(MAX_CANDIDATES, max(24, limit * 8))
    exclusion = (
        "AND c.id <> CAST(:exclude_claim_id AS uuid)" if exclude_claim_id else ""
    )
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query_text.casefold())))[:64]
    params: dict[str, Any] = {
        "embedding": vector_literal(embedding) if embedding is not None else None,
        "lexical_query": " OR ".join(terms),
        "cutoff": as_utc(cutoff),
        "historical": historical,
        "candidate_limit": candidate_limit,
        "content_cap": MAX_CONTENT_CHARS,
    }
    if exclude_claim_id:
        params["exclude_claim_id"] = exclude_claim_id
    eligible = f"""
        c.extracted_at <= :cutoff AND a.collected_at <= :cutoff
        AND (a.published_at <= :cutoff
             OR (:historical = false AND a.published_at IS NULL))
        {exclusion}
    """
    result = await db.execute(
        text(f"""
            WITH vector_hits AS (
                SELECT c.id, 1 - (c.embedding <=> CAST(:embedding AS vector)) AS similarity,
                       row_number() OVER (ORDER BY c.embedding <=> CAST(:embedding AS vector), c.id) AS rank
                FROM claims c JOIN articles a ON a.id = c.article_id
                WHERE c.embedding IS NOT NULL AND CAST(:embedding AS vector) IS NOT NULL
                  AND {eligible}
                ORDER BY c.embedding <=> CAST(:embedding AS vector), c.id
                LIMIT :candidate_limit
            ), keyword_hits AS (
                SELECT c.id,
                       ts_rank_cd(to_tsvector('english', c.claim_text),
                                  websearch_to_tsquery('english', :lexical_query)) AS lexical_score,
                       row_number() OVER (ORDER BY
                           ts_rank_cd(to_tsvector('english', c.claim_text),
                                      websearch_to_tsquery('english', :lexical_query)) DESC, c.id) AS rank
                FROM claims c JOIN articles a ON a.id = c.article_id
                WHERE to_tsvector('english', c.claim_text)
                          @@ websearch_to_tsquery('english', :lexical_query)
                  AND {eligible}
                ORDER BY lexical_score DESC, c.id
                LIMIT :candidate_limit
            ), fused AS (
                SELECT COALESCE(v.id, k.id) AS id, v.similarity,
                       COALESCE(k.lexical_score, 0) AS lexical_score,
                       COALESCE(1.0 / (60 + v.rank), 0)
                         + COALESCE(1.0 / (60 + k.rank), 0) AS retrieval_score
                FROM vector_hits v FULL OUTER JOIN keyword_hits k ON k.id = v.id
            )
            SELECT c.id::text AS claim_id, c.article_id::text AS article_id,
                   c.claim_text AS text, c.extracted_at,
                   a.url, a.title, a.domain, a.published_at, a.collected_at,
                   a.canonical_article_id::text AS canonical_article_id,
                   a.content_hash,
                   LEFT(a.cleaned_content, :content_cap) AS cleaned_content,
                   LEFT(a.content, :content_cap) AS content,
                   f.similarity, f.lexical_score, f.retrieval_score
            FROM fused f JOIN claims c ON c.id = f.id
            JOIN articles a ON a.id = c.article_id
            ORDER BY f.retrieval_score DESC, c.id
            LIMIT :candidate_limit
        """),
        params,
    )
    candidates: list[EvidenceCandidate] = []
    warnings: list[str] = []
    for row in result.mappings().all()[:candidate_limit]:
        try:
            candidates.append(_candidate(dict(row)))
        except (KeyError, TypeError, ValueError, OverflowError):
            logger.warning("Skipping malformed corpus evidence candidate")
            if not warnings:
                warnings.append("Some malformed corpus evidence records were excluded.")
    return candidates, warnings


def _origin_tokens(candidate: EvidenceCandidate) -> set[str]:
    tokens = {f"article:{candidate.article_id}"}
    if candidate.canonical_article_id:
        tokens.add(f"article:{candidate.canonical_article_id}")
    if candidate.content_hash:
        tokens.add(f"hash:{candidate.content_hash}")
    body = candidate.cleaned_content or candidate.content or ""
    normalized = " ".join(body.split()).casefold()
    if len(normalized) >= 80:
        tokens.add(f"body:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}")
    return tokens


def group_candidates(
    candidates: Sequence[EvidenceCandidate],
) -> list[EvidenceCandidate]:
    """Union canonical links and exact article-content duplicates, not claim wording."""
    parent = list(range(len(candidates)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        for token in _origin_tokens(candidate):
            if token in seen:
                parent[root(index)] = root(seen[token])
            else:
                seen[token] = index
    groups: dict[int, list[EvidenceCandidate]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(root(index), []).append(candidate)
    names: dict[int, str] = {}
    for index, group in groups.items():
        canonical_ids = sorted(
            {c.canonical_article_id for c in group if c.canonical_article_id}
        )
        hashes = sorted({c.content_hash for c in group if c.content_hash})
        if canonical_ids:
            names[index] = f"canonical:{canonical_ids[0]}"
        elif hashes:
            names[index] = f"content:{hashes[0]}"
        else:
            names[index] = f"article:{min(c.article_id for c in group)}"
    return [
        replace(c, syndication_group=names[root(i)]) for i, c in enumerate(candidates)
    ]


def select_candidates(
    candidates: Sequence[EvidenceCandidate],
    limit: int,
    *,
    own_domain: str | None = None,
    own_origin: EvidenceCandidate | None = None,
) -> list[EvidenceCandidate]:
    """Deduplicate origins, then round-robin domains in similarity-ranked order."""
    own_domain = canonical_domain(own_domain)
    own_tokens = _origin_tokens(own_origin) if own_origin else set()
    grouped = group_candidates([*candidates, *([own_origin] if own_origin else [])])
    own_group = grouped[-1].syndication_group if own_origin else None
    ranked = sorted(
        grouped[: len(candidates)],
        key=lambda c: (
            -(c.retrieval_score if c.retrieval_score is not None else (c.similarity or 0)),
            c.claim_id,
        ),
    )
    by_domain: OrderedDict[str | None, list[EvidenceCandidate]] = OrderedDict()
    seen_groups: set[str] = set()
    for candidate in ranked:
        if (
            (own_domain and candidate.domain == own_domain)
            or (own_group and candidate.syndication_group == own_group)
            or (_origin_tokens(candidate) & own_tokens)
            or candidate.syndication_group in seen_groups
        ):
            continue
        seen_groups.add(candidate.syndication_group)
        by_domain.setdefault(candidate.domain, []).append(candidate)
    selected: list[EvidenceCandidate] = []
    while by_domain and len(selected) < limit:
        for domain in list(by_domain):
            selected.append(by_domain[domain].pop(0))
            if not by_domain[domain]:
                del by_domain[domain]
            if len(selected) == limit:
                break
    return selected


def evidence_passage(
    candidate: EvidenceCandidate, claim_text: str
) -> tuple[str, PassageSource]:
    """Return a contiguous stored-body excerpt, or explicitly sourced extracted text."""
    if candidate.cleaned_content and candidate.cleaned_content.strip():
        body, source = candidate.cleaned_content, "cleaned_content"
    elif candidate.content and candidate.content.strip():
        body, source = candidate.content, "content"
    else:
        return candidate.text, "extracted_claim"
    body = body[:MAX_CONTENT_CHARS]
    spans = list(re.finditer(r"\S.*?(?:[.!?](?=\s|$)|\n|$)", body, re.DOTALL))
    if not spans:
        return candidate.text, "extracted_claim"
    anchor = (
        re.search(re.escape(candidate.text), body, re.IGNORECASE)
        if candidate.text
        else None
    )
    terms = set(re.findall(r"\w{3,}", f"{candidate.text} {claim_text}".casefold()))
    if anchor:
        index = next(
            (i for i, span in enumerate(spans) if span.end() > anchor.start()), 0
        )
    else:
        index = max(
            range(len(spans)),
            key=lambda i: len(
                terms & set(re.findall(r"\w{3,}", spans[i].group().casefold()))
            ),
        )
    start, end = spans[index].span()
    if end - start > MAX_PASSAGE_CHARS:
        start = max(start, (anchor.start() - 200) if anchor else start)
        end = min(len(body), start + MAX_PASSAGE_CHARS)
    else:
        # Adjacent context helps avoid presenting an attributed/denied quote alone.
        if index and end - spans[index - 1].start() <= MAX_PASSAGE_CHARS:
            start = spans[index - 1].start()
        if (
            index + 1 < len(spans)
            and spans[index + 1].end() - start <= MAX_PASSAGE_CHARS
        ):
            end = spans[index + 1].end()
    passage_source: PassageSource = (
        "cleaned_content" if source == "cleaned_content" else "content"
    )
    return body[start:end].strip(), passage_source


def assessment_for(evidence: Sequence[EvidenceItem]) -> Assessment:
    support = any(item["stance"] == "entailment" for item in evidence)
    contradiction = any(item["stance"] == "contradiction" for item in evidence)
    if support and contradiction:
        return "MIXED_EVIDENCE"
    if support:
        return "SUPPORTED_BY_CORPUS"
    if contradiction:
        return "CONTRADICTED_BY_CORPUS"
    return "INSUFFICIENT_EVIDENCE"


async def check_claim_with_embedding(
    db: AsyncSession,
    claim_text: str,
    embedding: Sequence[float] | str | None,
    *,
    as_of: datetime | None = None,
    limit: int = 6,
    observed_at: datetime | None = None,
    exclude_claim_id: str | None = None,
    own_domain: str | None = None,
    own_origin: EvidenceCandidate | None = None,
) -> ClaimCheckResult:
    """Shared read-only checking for API inputs and already-embedded batch claims."""
    if type(limit) is not int or not 1 <= limit <= MAX_EVIDENCE:
        raise ValueError("Evidence limit must be an integer between 1 and 12")
    observed = as_utc(observed_at or datetime.now(UTC))
    cutoff = min(as_utc(as_of), observed) if as_of else observed
    warnings = [CORPUS_WARNING, INDEPENDENCE_WARNING]
    if as_of:
        warnings.append(
            "Historical mode excludes unknown publication times and evidence "
            "published, collected or extracted after as_of. Stored content is "
            "not a reconstruction of historical article revisions."
        )
        if as_of.tzinfo is None:
            warnings.append("as_of has no timezone offset; it is interpreted as UTC.")
        if as_utc(as_of) > observed:
            warnings.append(
                "Future as_of was capped at observed_at; future evidence is excluded."
            )
    candidates, retrieval_warnings = await retrieve_candidates(
        db,
        embedding,
        cutoff=cutoff,
        historical=as_of is not None,
        limit=limit,
        exclude_claim_id=exclude_claim_id,
        query_text=claim_text,
    )
    if embedding is None:
        warnings.append(
            "The embedding model is unavailable; keyword retrieval was used. "
            "Semantic matches without shared terms may be missed."
        )
    warnings.extend(retrieval_warnings)
    minimum = near_window()[0]
    eligible: list[EvidenceCandidate] = []
    for candidate in candidates:
        if (
            candidate.lexical_score <= 0
            and (candidate.similarity is None or candidate.similarity < minimum)
        ) or not candidate.text.strip():
            continue
        timestamps = [
            candidate.published_at,
            candidate.collected_at,
            candidate.extracted_at,
        ]
        if any(t is not None and t > cutoff for t in timestamps):
            continue
        if as_of and any(t is None for t in timestamps):
            continue
        eligible.append(candidate)
    selected = select_candidates(
        eligible, limit, own_domain=own_domain, own_origin=own_origin
    )
    evidence: list[EvidenceItem] = []
    for candidate in selected:
        passage, source = evidence_passage(candidate, claim_text)
        stance = await classify_stance(claim_text, passage)
        if stance.abstention_reason:
            warnings.append(
                f"NLI abstained: {stance.abstention_reason.replace('_', ' ')}."
            )
        if source == "extracted_claim":
            warnings.append(
                "Some evidence uses extracted-claim text (passage_source='extracted_claim'), "
                "not an article quotation; the extraction itself is unverified."
            )
        if candidate.published_at is None:
            warnings.append(
                "Some current-corpus evidence has unknown publication time."
            )
        if candidate.domain is None:
            warnings.append(
                "Some evidence lacks a domain; source independence cannot be assessed."
            )
        body = (
            candidate.cleaned_content if source == "cleaned_content"
            else candidate.content if source == "content" else None
        )
        start = body.find(passage) if body is not None else -1
        evidence.append(
            {
                "claim_id": candidate.claim_id,
                "article_id": candidate.article_id,
                "text": candidate.text,
                "passage": passage,
                "passage_source": source,
                "url": candidate.url,
                "title": candidate.title,
                "domain": candidate.domain,
                "published_at": candidate.published_at,
                "similarity": round(candidate.similarity, 6) if candidate.similarity is not None else None,
                "stance": stance.stance,
                "syndication_group": candidate.syndication_group,
                "retrieval_method": candidate.retrieval_method,
                "collected_at": candidate.collected_at,
                "extracted_at": candidate.extracted_at,
                "passage_start": start if start >= 0 else None,
                "passage_end": start + len(passage) if start >= 0 else None,
                "passage_sha256": hashlib.sha256(passage.encode("utf-8")).hexdigest(),
                "abstention_reason": stance.abstention_reason,
            }
        )
    if not evidence:
        warnings.append("No eligible evidence was found in the bounded corpus search.")
    elif all(item["stance"] == "neutral" for item in evidence):
        warnings.append(
            "Retrieved evidence did not establish entailment or contradiction."
        )
    return {
        "claim_text": claim_text,
        "assessment": assessment_for(evidence),
        "evidence": evidence,
        "warnings": list(dict.fromkeys(warnings)),
        "score_kind": SCORE_KIND,
        "observed_at": observed,
        "method_version": METHOD_VERSION,
    }


async def check_claim(
    db: AsyncSession,
    claim_text: str,
    *,
    as_of: datetime | None = None,
    limit: int = 6,
) -> ClaimCheckResult:
    embedding: list[float] | None
    try:
        embedding = await embed_query(claim_text)
    except EmbeddingUnavailableError:
        logger.info("Embedding unavailable; using disclosed keyword corpus retrieval")
        embedding = None
    return await check_claim_with_embedding(
        db, claim_text, embedding, as_of=as_of, limit=limit
    )
