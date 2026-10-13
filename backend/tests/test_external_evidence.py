"""Fixed-source revision retrieval with mocked HTTP and NLI; no live research claims."""

import hashlib
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.verdict import external
from app.services.verdict.engine import NLIUnavailableError, StanceResult
from tests.test_evidence_check import NOW, PAST, QUERY


def page(revision_id=123, **changes) -> dict[str, Any]:
    result = {
        "pageid": revision_id,
        "title": "Council decision",
        "revisions": [{"revid": revision_id, "timestamp": PAST.isoformat()}],
    }
    result.update(changes)
    return result


def transport_for(pages, *, parsed=None, requests=None) -> httpx.MockTransport:
    def handler(request):
        if requests is not None:
            requests.append(request)
        assert request.url.host == "en.wikipedia.org"
        if request.url.params["action"] == "query":
            return httpx.Response(200, json={"batchcomplete": True, "query": {"pages": pages}})
        revision = int(request.url.params["oldid"])
        payload = parsed if parsed is not None else {
            "parse": {"revid": revision, "text": f"<p>{QUERY} Construction starts next year.</p>"}
        }
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_revision_passage_exact_provenance_and_local_stance(monkeypatch):
    classify = AsyncMock(return_value=StanceResult("entailment"))
    monkeypatch.setattr(external, "classify_stance", classify)
    rendered = f"<script>untrusted script</script><p>{QUERY} Construction starts next year.</p>"
    async with httpx.AsyncClient(transport=transport_for(
        [page()], parsed={"parse": {"revid": 123, "text": rendered}}
    )) as client:
        result = await external.check_external_claim(QUERY, client=client, observed_at=NOW)
    item = result["sources"][0]
    assert result["assessment"] == result["verdict"] == "SUPPORTED_BY_SOURCES"
    assert item["url"] == "https://en.wikipedia.org/w/index.php?oldid=123"
    assert item["revision_id"] == 123
    assert item["published_at"] == PAST
    assert item["retrieved_at"] == NOW
    body = external._plain_text(rendered)
    assert body[item["passage_start"] : item["passage_end"]] == item["passage"]
    assert "script" not in item["passage"]
    assert item["passage_sha256"] == hashlib.sha256(item["passage"].encode()).hexdigest()
    assert item["source_group"] == "wikipedia"
    classify.assert_awaited_once_with(QUERY, item["passage"])


@pytest.mark.asyncio
async def test_historical_query_and_future_revision_exclusion(monkeypatch):
    classify = AsyncMock(return_value=StanceResult("neutral"))
    monkeypatch.setattr(external, "classify_stance", classify)
    requests: list[httpx.Request] = []
    pages = [page(), page(124, revisions=[{
        "revid": 124, "timestamp": (PAST + timedelta(seconds=1)).isoformat()
    }])]
    async with httpx.AsyncClient(transport=transport_for(pages, requests=requests)) as client:
        result = await external.check_external_claim(
            QUERY, as_of=PAST.replace(tzinfo=None), client=client, observed_at=NOW
        )
    assert len(result["sources"]) == 1
    assert len(requests) == 2  # Search + one eligible revision; future revision never fetched.
    params = requests[0].url.params
    assert params["rvdir"] == "older"
    assert params["rvstart"] == "2024-01-01T00:00:00Z"
    assert any("interpreted as UTC" in warning for warning in result["warnings"])
    assert any("current search index" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_future_as_of_is_capped_before_source_request(monkeypatch):
    requests: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=transport_for([], requests=requests)) as client:
        result = await external.check_external_claim(
            QUERY, as_of=NOW + timedelta(days=1), client=client, observed_at=NOW
        )
    assert requests[0].url.params["rvstart"] == NOW.isoformat().replace("+00:00", "Z")
    assert any("capped" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_empty_source_search_does_not_load_model(monkeypatch):
    classify = AsyncMock(side_effect=AssertionError("No model needed"))
    monkeypatch.setattr(external, "classify_stance", classify)
    async with httpx.AsyncClient(transport=transport_for([])) as client:
        result = await external.check_external_claim(QUERY, client=client, observed_at=NOW)
    assert result["assessment"] == "UNVERIFIABLE"
    assert result["sources"] == []
    classify.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_failure_keeps_retrieved_passages_with_explicit_abstention(monkeypatch):
    classify = AsyncMock(side_effect=NLIUnavailableError("Pre-cache the model"))
    monkeypatch.setattr(external, "classify_stance", classify)
    async with httpx.AsyncClient(transport=transport_for([page(), page(124)])) as client:
        result = await external.check_external_claim(QUERY, client=client, observed_at=NOW)
    assert result["assessment"] == "UNVERIFIABLE"
    assert len(result["sources"]) == 2
    assert all(item["abstention_reason"] == "model_unavailable" for item in result["sources"])
    assert all(item["stance"] == "neutral" for item in result["sources"])
    classify.assert_awaited_once()


@pytest.mark.asyncio
async def test_opposing_passages_produce_mixed_evidence(monkeypatch):
    monkeypatch.setattr(external, "classify_stance", AsyncMock(side_effect=[
        StanceResult("entailment"), StanceResult("contradiction"),
    ]))
    async with httpx.AsyncClient(transport=transport_for([page(), page(124)])) as client:
        result = await external.check_external_claim(QUERY, client=client, observed_at=NOW)
    assert result["assessment"] == "MIXED_EVIDENCE"
    assert any("not independent corroboration" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_requested_limit_and_duplicate_revisions_are_bounded(monkeypatch):
    classify = AsyncMock(return_value=StanceResult("neutral"))
    monkeypatch.setattr(external, "classify_stance", classify)
    requests: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=transport_for([page(i) for i in range(1, 8)], requests=requests)) as client:
        result = await external.check_external_claim(QUERY, limit=1, client=client, observed_at=NOW)
    assert len(requests) == 2
    assert len(result["sources"]) == 1
    assert requests[0].url.params["gsrlimit"] == "1"
    async with httpx.AsyncClient(transport=transport_for([page(), page(), page(124)])) as client:
        result = await external.check_external_claim(QUERY, client=client, observed_at=NOW)
    assert [item["revision_id"] for item in result["sources"]] == [123, 124]


@pytest.mark.asyncio
@pytest.mark.parametrize("parsed", [
    {"parse": {"revid": 456, "text": "wrong revision"}},
    {"parse": {"text": "no revision identity"}},
    {"parse": {"revid": 123, "text": None}},
    {"error": {"code": "badrevids"}},
])
async def test_invalid_revision_response_never_claims_a_valid_source(parsed):
    async with httpx.AsyncClient(transport=transport_for([page()], parsed=parsed)) as client:
        with pytest.raises(external.ExternalEvidenceUnavailableError):
            await external.check_external_claim(QUERY, client=client, observed_at=NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["redirect", "oversized", "invalid_json", "timeout", "http_error"])
async def test_upstream_failures_are_bounded_and_actionable(kind):
    calls = []

    def handler(request):
        calls.append(request)
        if kind == "redirect":
            return httpx.Response(302, headers={"Location": "https://unrelated.example"})
        if kind == "oversized":
            return httpx.Response(200, content=b"x" * (external.MAX_RESPONSE_BYTES + 1))
        if kind == "invalid_json":
            return httpx.Response(200, content=b"not JSON")
        if kind == "timeout":
            raise httpx.ReadTimeout("private upstream detail", request=request)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        with pytest.raises(external.ExternalEvidenceUnavailableError) as error:
            await external.check_external_claim(QUERY, client=client, observed_at=NOW)
    assert len(calls) == 1
    assert "private upstream" not in str(error.value)


@pytest.mark.asyncio
async def test_partial_revision_failure_preserves_available_source_with_warning(monkeypatch):
    monkeypatch.setattr(external, "classify_stance", AsyncMock(return_value=StanceResult("neutral")))
    handler = transport_for([page(), page(124)])

    async def partial(request):
        if request.url.params.get("oldid") == "124":
            return httpx.Response(503)
        return await handler.handle_async_request(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(partial)) as client:
        result = await external.check_external_claim(QUERY, client=client, observed_at=NOW)
    assert len(result["sources"]) == 1
    assert any("coverage is incomplete" in warning for warning in result["warnings"])
