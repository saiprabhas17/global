"""Revision-bound Wikipedia passages with local textual stance classification.

Only the fixed Wikipedia Action API is contacted. Links identify the retrieved
revision, not a mutable search snippet, and model failures remain abstentions.
This is bounded source investigation, never a probability that a claim is true.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

import httpx
from bs4 import BeautifulSoup

from app.services.verdict.engine import (
    NLIUnavailableError,
    Stance,
    StanceResult,
    classify_stance,
)
from app.services.verdict.evidence import (
    MAX_CONTENT_CHARS,
    EvidenceCandidate,
    as_utc,
    evidence_passage,
)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
METHOD_VERSION = "wikipedia-revision-passage-nli-v1"
MAX_PAGES = 3
MAX_RESPONSE_BYTES = 2_000_000

ExternalAssessment = Literal[
    "SUPPORTED_BY_SOURCES", "CONTRADICTED_BY_SOURCES", "MIXED_EVIDENCE", "UNVERIFIABLE"
]


class ExternalEvidenceUnavailableError(RuntimeError):
    """The fixed upstream source is temporarily unavailable or returned invalid data."""


class ExternalSource(TypedDict):
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


class ExternalCheckResult(TypedDict):
    claim: str
    verdict: ExternalAssessment
    assessment: ExternalAssessment
    explanation: str
    sources: list[ExternalSource]
    observed_at: datetime
    method_version: str
    warnings: list[str]


async def _api_json(client: httpx.AsyncClient, params: dict[str, str | int]) -> dict[str, Any]:
    """Bound response bytes and time; reject redirects and MediaWiki API errors."""
    try:
        async with client.stream("GET", WIKIPEDIA_API, follow_redirects=False, params={
            "format": "json", "formatversion": 2, **params,
        }) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise ExternalEvidenceUnavailableError(
                        "Wikipedia returned an oversized response; retry a more specific claim."
                    )
        payload = json.loads(data)
        if not isinstance(payload, dict) or "error" in payload:
            raise ValueError("Invalid Wikipedia API result")
        return payload
    except ExternalEvidenceUnavailableError:
        raise
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise ExternalEvidenceUnavailableError(
            "Wikipedia evidence is temporarily unavailable. Check network access and retry."
        ) from exc


def _revision_pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    query = payload.get("query")
    if not isinstance(query, dict):
        # An empty search has no query object; malformed non-null values are errors.
        if query is None:
            return []
        raise ExternalEvidenceUnavailableError("Wikipedia returned invalid search results.")
    pages = query.get("pages", [])
    if not isinstance(pages, list):
        raise ExternalEvidenceUnavailableError("Wikipedia returned invalid search results.")
    return [page for page in pages if isinstance(page, dict)][:MAX_PAGES]


def _plain_text(rendered: str) -> str:
    soup = BeautifulSoup(rendered, "html.parser")
    for node in soup.select("script, style, .mw-editsection, .reference, .reflist, .navbox, .metadata"):
        node.decompose()
    # Keep a deterministic source representation for offsets and hashes.
    return " ".join(soup.get_text(" ", strip=True).split())[:MAX_CONTENT_CHARS]


async def _page_passage(
    client: httpx.AsyncClient,
    page: dict[str, Any],
    claim_text: str,
    cutoff: datetime,
    observed: datetime,
) -> ExternalSource | None:
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or not revisions or not isinstance(revisions[0], dict):
        return None
    revision = revisions[0]
    revision_id = revision.get("revid")
    title = page.get("title")
    timestamp = revision.get("timestamp")
    if type(revision_id) is not int or revision_id <= 0 or not isinstance(title, str):
        return None
    try:
        published = as_utc(datetime.fromisoformat(str(timestamp)))
    except (ValueError, TypeError):
        return None
    if published > cutoff:
        return None
    payload = await _api_json(client, {
        "action": "parse", "oldid": revision_id, "prop": "text", "disableeditsection": 1,
    })
    parsed = payload.get("parse")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("text"), str):
        raise ExternalEvidenceUnavailableError("Wikipedia returned an invalid revision passage.")
    # MediaWiki returns the resolved revision ID; reject an unexpected revision.
    if parsed.get("revid") != revision_id:
        raise ExternalEvidenceUnavailableError("Wikipedia returned a different source revision.")
    body = _plain_text(parsed["text"])
    if not body.strip():
        return None
    url = f"https://en.wikipedia.org/w/index.php?oldid={revision_id}"
    candidate = EvidenceCandidate(
        claim_id="", article_id=str(page.get("pageid", "")), text="", url=url,
        title=title, domain="en.wikipedia.org", published_at=published,
        similarity=None, cleaned_content=body,
    )
    passage, _ = evidence_passage(candidate, claim_text)
    start = body.find(passage)
    if not passage or start < 0:
        return None
    return {
        "title": title, "url": url, "passage": passage, "stance": "neutral",
        "abstention_reason": None, "revision_id": revision_id,
        "published_at": published, "retrieved_at": observed,
        "passage_start": start, "passage_end": start + len(passage),
        "passage_sha256": hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        "source_group": "wikipedia",
    }


async def check_external_claim(
    claim_text: str,
    *,
    as_of: datetime | None = None,
    limit: int = MAX_PAGES,
    client: httpx.AsyncClient | None = None,
    observed_at: datetime | None = None,
) -> ExternalCheckResult:
    """Retrieve at most three revision passages, then classify only their exact text."""
    if type(limit) is not int or not 1 <= limit <= 12:
        raise ValueError("Evidence limit must be an integer between 1 and 12")
    observed = as_utc(observed_at or datetime.now(UTC))
    cutoff = min(as_utc(as_of), observed) if as_of else observed
    params: dict[str, str | int] = {
        "action": "query", "generator": "search", "gsrsearch": claim_text,
        "gsrnamespace": 0, "gsrlimit": min(MAX_PAGES, limit),
        "prop": "revisions", "rvprop": "ids|timestamp", "rvlimit": 1,
        "rvstart": cutoff.isoformat().replace("+00:00", "Z"), "rvdir": "older",
    }
    warnings = [
        "This check compares a claim with bounded Wikipedia passages, not verified truth.",
        "All retrieved pages belong to one source group (Wikipedia); they are not independent corroboration.",
        "Passage offsets refer to normalized visible text from the linked revision. Revision time is not event time.",
        "Retrieval examines at most three pages and the first 32,000 characters of each rendered revision.",
    ]
    if as_of:
        warnings.append(
            "Historical mode retrieves revisions at or before as_of using Wikipedia's current search index; "
            "it does not reconstruct historical search rankings or historical transcluded templates."
        )
        if as_of.tzinfo is None:
            warnings.append("as_of has no timezone offset; it is interpreted as UTC.")
        if as_utc(as_of) > observed:
            warnings.append("Future as_of was capped at observed_at.")

    async def fetch(active_client: httpx.AsyncClient) -> list[ExternalSource]:
        payload = await _api_json(active_client, params)
        pages = _revision_pages(payload)[:limit]
        results = await asyncio.gather(*[
            _page_passage(active_client, page, claim_text, cutoff, observed)
            for page in pages
        ], return_exceptions=True)
        sources: list[ExternalSource] = []
        failures: list[ExternalEvidenceUnavailableError] = []
        seen_revisions: set[int] = set()
        for result in results:
            if isinstance(result, ExternalEvidenceUnavailableError):
                failures.append(result)
            elif isinstance(result, BaseException):
                raise result
            elif result is not None and result["revision_id"] not in seen_revisions:
                sources.append(result)
                seen_revisions.add(result["revision_id"])
        if failures:
            if not sources:
                raise failures[0]
            warnings.append("Some source revisions could not be retrieved; coverage is incomplete.")
        return sources

    if client is None:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(12.0), follow_redirects=False,
            headers={"User-Agent": "GMEE/0.1 (academic claim evidence investigation)"},
        ) as owned_client:
            sources = await fetch(owned_client)
    else:
        sources = await fetch(client)

    unavailable: str | None = None
    for source in sources:
        if unavailable:
            stance = StanceResult("neutral", "model_unavailable")
        else:
            try:
                stance = await classify_stance(claim_text, source["passage"])
            except NLIUnavailableError as exc:
                unavailable = str(exc)
                stance = StanceResult("neutral", "model_unavailable")
                warnings.append(f"Stance classification is unavailable: {exc}")
        source["stance"] = stance.stance
        source["abstention_reason"] = stance.abstention_reason
        if stance.abstention_reason:
            warnings.append(f"NLI abstained: {stance.abstention_reason.replace('_', ' ')}.")
    support = any(source["stance"] == "entailment" for source in sources)
    contradiction = any(source["stance"] == "contradiction" for source in sources)
    assessment: ExternalAssessment = (
        "MIXED_EVIDENCE" if support and contradiction else
        "SUPPORTED_BY_SOURCES" if support else
        "CONTRADICTED_BY_SOURCES" if contradiction else "UNVERIFIABLE"
    )
    explanations = {
        "MIXED_EVIDENCE": "Retrieved passages contain both supporting and contradicting text. Review the cited context.",
        "SUPPORTED_BY_SOURCES": "The local stance model found support in the cited passages. This does not independently establish truth.",
        "CONTRADICTED_BY_SOURCES": "The local stance model found a contradiction in the cited passages. Review the revision and claim context.",
        "UNVERIFIABLE": "The retrieved passages did not establish support or contradiction; absence of evidence is not falsity.",
    }
    if not sources:
        warnings.append("No eligible revision passages were found in the bounded Wikipedia search.")
    return {
        "claim": claim_text, "verdict": assessment, "assessment": assessment,
        "explanation": explanations[assessment], "sources": sources,
        "observed_at": observed, "method_version": METHOD_VERSION,
        "warnings": list(dict.fromkeys(warnings)),
    }
