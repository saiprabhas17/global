import uuid
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.models.article import Article
from app.models.source import Source, SourceTypeEnum
from app.models.user import RoleEnum, User
from app.services.collection_orchestrator import CollectionOrchestrator
from app.services.collectors.base import MissingCredentialsError
from app.services.collectors.news_api import NewsAPICollector
from app.services.collectors.reddit import RedditCollector
from app.services.collectors.rss import RSSCollector


class MockResponse:
    def __init__(
        self,
        text: str = "",
        json_data: Any = None,
        status_code: int = 200,
    ) -> None:
        self.text = text
        self._json = json_data
        self.status_code = status_code

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("Error", request=None, response=self)  # type: ignore


@pytest.fixture
def mock_httpx_get(monkeypatch):
    original_get = httpx.AsyncClient.get
    
    async def mock_get(self, url, *args, **kwargs):
        url_str = str(url)
        if "newsapi.org" in url_str:
            return MockResponse(json_data={
                "status": "ok",
                "articles": [
                    {"title": "News 1", "url": "http://news.com/1", "content": "c1", "publishedAt": "2026-01-01T00:00:00Z"},
                    {"title": "[Removed]", "url": "http://news.com/removed"}
                ]
            })
        if "oauth.reddit.com" in url_str:
            return MockResponse(json_data={
                "data": {
                    "children": [
                        {"data": {"title": "Post 1", "url": "http://reddit.com/1", "selftext": "text1", "created_utc": 1700000000}},
                    ]
                }
            })
        if "bbci.co.uk" in url_str:
            return MockResponse(text="""<?xml version="1.0" encoding="UTF-8"?>
                <rss><channel>
                    <item><title>RSS 1</title><link>http://rss.com/1</link><description>rss1</description></item>
                </channel></rss>
            """)
        return await original_get(self, url, *args, **kwargs)
        
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

@pytest.fixture
def mock_httpx_post(monkeypatch):
    original_post = httpx.AsyncClient.post
    
    async def mock_post(self, url, *args, **kwargs):
        if "access_token" in str(url):
            return MockResponse(json_data={"access_token": "mock_token", "expires_in": 3600})
        return await original_post(self, url, *args, **kwargs)
        
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)


async def test_rss_collector(mock_httpx_get):
    collector = RSSCollector()
    source = Source(id=uuid.uuid4(), name="BBC", type=SourceTypeEnum.rss, url_or_identifier="http://feeds.bbci.co.uk/news")
    articles = await collector.collect(source)
    assert len(articles) == 1
    assert articles[0].title == "RSS 1"


async def test_rss_collector_malformed(monkeypatch):
    async def mock_get(*args, **kwargs):
        return MockResponse(text="not xml")
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    
    collector = RSSCollector()
    source = Source(id=uuid.uuid4(), name="BBC", type=SourceTypeEnum.rss, url_or_identifier="http://test.com")
    articles = await collector.collect(source)
    assert len(articles) == 0


async def test_newsapi_collector_success(mock_httpx_get, monkeypatch):
    collector = NewsAPICollector()
    monkeypatch.setattr(collector.settings, "NEWSAPI_KEY", "mock_key")
    source = Source(id=uuid.uuid4(), name="News", type=SourceTypeEnum.news_api, url_or_identifier="test")
    articles = await collector.collect(source)
    assert len(articles) == 1  # [Removed] should be skipped
    assert articles[0].title == "News 1"


async def test_newsapi_collector_no_key(mock_httpx_get, monkeypatch):
    collector = NewsAPICollector()
    monkeypatch.setattr(collector.settings, "NEWSAPI_KEY", "")
    source = Source(id=uuid.uuid4(), name="News", type=SourceTypeEnum.news_api, url_or_identifier="test")
    with pytest.raises(MissingCredentialsError):
        await collector.collect(source)


async def test_reddit_collector_success(mock_httpx_get, mock_httpx_post, monkeypatch):
    collector = RedditCollector()
    monkeypatch.setattr(collector.settings, "REDDIT_CLIENT_ID", "mock_id")
    monkeypatch.setattr(collector.settings, "REDDIT_CLIENT_SECRET", "mock_secret")
    source = Source(id=uuid.uuid4(), name="Reddit", type=SourceTypeEnum.reddit, url_or_identifier="test")
    articles = await collector.collect(source)
    assert len(articles) == 1
    assert articles[0].title == "Post 1"


async def test_reddit_collector_no_creds(mock_httpx_get, mock_httpx_post, monkeypatch):
    collector = RedditCollector()
    monkeypatch.setattr(collector.settings, "REDDIT_CLIENT_ID", "")
    source = Source(id=uuid.uuid4(), name="Reddit", type=SourceTypeEnum.reddit, url_or_identifier="test")
    with pytest.raises(MissingCredentialsError):
        await collector.collect(source)


async def test_orchestrator_dedup(db_session: AsyncSession, mock_httpx_get, mock_httpx_post, monkeypatch):
    # Insert a source
    source = Source(name="BBC", type=SourceTypeEnum.rss, url_or_identifier="http://feeds.bbci.co.uk/news", is_active=True)
    db_session.add(source)
    await db_session.commit()

    orchestrator = CollectionOrchestrator()
    
    # Run cycle 1
    summaries1 = await orchestrator.run_collection_cycle(db_session)
    assert summaries1[0].articles_inserted == 1
    
    # Run cycle 2
    summaries2 = await orchestrator.run_collection_cycle(db_session)
    assert summaries2[0].articles_inserted == 0  # Dedup prevents insertion

    result = await db_session.execute(select(Article))
    assert len(result.scalars().all()) == 1


async def test_orchestrator_resilience(db_session: AsyncSession, mock_httpx_get, mock_httpx_post, monkeypatch):
    # Insert 3 sources
    s1 = Source(name="BBC", type=SourceTypeEnum.rss, url_or_identifier="http://feeds.bbci.co.uk/news", is_active=True)
    s2 = Source(name="News", type=SourceTypeEnum.news_api, url_or_identifier="test", is_active=True)
    s3 = Source(name="Reddit", type=SourceTypeEnum.reddit, url_or_identifier="test", is_active=True)
    db_session.add_all([s1, s2, s3])
    await db_session.commit()

    orchestrator = CollectionOrchestrator()
    # Missing creds for NewsAPI
    monkeypatch.setattr(orchestrator.collectors[SourceTypeEnum.news_api].settings, "NEWSAPI_KEY", "")
    monkeypatch.setattr(orchestrator.collectors[SourceTypeEnum.reddit].settings, "REDDIT_CLIENT_ID", "mock_id")
    monkeypatch.setattr(orchestrator.collectors[SourceTypeEnum.reddit].settings, "REDDIT_CLIENT_SECRET", "mock_sec")

    # Mock Reddit collector to crash unexpectedly
    async def crash_collect(*args, **kwargs):
        raise ValueError("Simulated crash")
    monkeypatch.setattr(orchestrator.collectors[SourceTypeEnum.reddit], "collect", crash_collect)

    summaries = await orchestrator.run_collection_cycle(db_session)
    
    # BBC should succeed, News should be skipped, Reddit should fail
    assert len(summaries) == 3
    
    bbc_sum = next(s for s in summaries if s.source_name == "BBC")
    news_sum = next(s for s in summaries if s.source_name == "News")
    reddit_sum = next(s for s in summaries if s.source_name == "Reddit")
    
    assert bbc_sum.error is None
    assert "Skipped - NewsAPI key missing" in (news_sum.error or "")
    assert "Failed - ValueError: Simulated crash" in (reddit_sum.error or "")


async def test_trigger_endpoint_admin(async_client, mock_httpx_get, mock_httpx_post, monkeypatch):
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: User(id=uuid.uuid4(), email="admin@test.com", role=RoleEnum.admin)
    
    response = await async_client.post("/api/v1/collection/trigger")
    
    app.dependency_overrides.pop(get_current_user, None)
    
    assert response.status_code == 200
    assert "summaries" in response.json()


async def test_trigger_endpoint_non_admin(async_client):
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: User(id=uuid.uuid4(), email="user@test.com", role=RoleEnum.user)
    
    response = await async_client.post("/api/v1/collection/trigger")
    
    app.dependency_overrides.pop(get_current_user, None)
    
    assert response.status_code == 403


async def test_status_endpoint_auth(async_client):
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: User(id=uuid.uuid4(), email="user@test.com", role=RoleEnum.user)
    
    response = await async_client.get("/api/v1/collection/status")
    
    app.dependency_overrides.pop(get_current_user, None)
    
    assert response.status_code == 200
    assert "sources" in response.json()
    assert "latest_run" in response.json()


async def test_status_endpoint_unauth(async_client):
    response = await async_client.get("/api/v1/collection/status")
    assert response.status_code == 401


@pytest.fixture(autouse=True)
def isolated_graph_projection(monkeypatch: pytest.MonkeyPatch):
    """Pipeline unit tests do not contact a live Neo4j projection."""
    from unittest.mock import AsyncMock

    from app.services.evolution.neo4j_writer import Neo4jWriter
    monkeypatch.setattr(Neo4jWriter, "sync_persisted_graph", AsyncMock(return_value={"success": True}))
