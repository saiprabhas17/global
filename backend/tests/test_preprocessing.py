import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.article import Article, ProcessingStatusEnum
from app.models.source import Source, SourceTypeEnum
from app.models.user import RoleEnum, User
from app.services.preprocessing.cleaner import clean_text
from app.services.preprocessing.language_detector import detect_language
from app.services.preprocessing.metadata_extractor import (
    compute_word_count,
    extract_domain,
    extract_or_repair_published_at,
)
from app.services.preprocessing.near_duplicate_detector import NearDuplicateDetector
from app.services.preprocessing_orchestrator import PreprocessingOrchestrator


def test_cleaner():
    html_text = "<p>This is   some <b>messy</b> text&amp;more.</p>\n\n  \n<br>Yes."
    cleaned = clean_text(html_text)
    assert cleaned == "This is some messy text&more.\nYes."
    
    assert clean_text(None) is None


def test_language_detector():
    english_text = "This is a sufficiently long english text that should be detected."
    assert detect_language(english_text) == "en"
    
    spanish_text = "Este es un texto lo suficientemente largo en español que debería ser detectado."
    assert detect_language(spanish_text) == "es"
    
    short_text = "Too short"
    assert detect_language(short_text) is None


def test_metadata_extractor():
    assert extract_domain("https://www.bbc.co.uk/news/world") == "www.bbc.co.uk"
    assert extract_domain(None) is None
    
    assert compute_word_count("One two three.") == 3
    
    # Test date repair
    dt = datetime(2026, 1, 1, tzinfo=UTC)
    
    # Already set
    assert extract_or_repair_published_at(dt, {}) == dt
    
    # Unix timestamp in created_utc
    repaired = extract_or_repair_published_at(None, {"created_utc": 1700000000})
    assert repaired is not None
    assert repaired.year == 2023
    
    # String in publishedAt
    repaired_str = extract_or_repair_published_at(None, {"publishedAt": "2026-05-01T12:00:00Z"})
    assert repaired_str is not None
    assert repaired_str.year == 2026


async def test_near_duplicate_detector(db_session: AsyncSession):
    # Setup source
    source = Source(name="Test", type=SourceTypeEnum.rss, url_or_identifier="test")
    db_session.add(source)
    await db_session.commit()
    
    now = datetime.now(UTC)
    
    base_text = " ".join([f"w{i:02d}" for i in range(1, 33)])
    below_text = " ".join([f"mod{k:02d}" if k in (1, 10) else f"w{k:02d}" for k in range(1, 33)])
    above_text = " ".join([f"mod{k:02d}" if k in (1, 4) else f"w{k:02d}" for k in range(1, 33)])

    art1 = Article(
        source_id=source.id,
        title="Title 1",
        url="http://test.com/1",
        content="raw",
        cleaned_content=base_text,
        content_hash="hash1",
        collected_at=now - timedelta(days=3)
    )
    
    db_session.add_all([art1])
    await db_session.commit()
    
    detector = NearDuplicateDetector(db_session)
    await detector.initialize()
    
    # Check what happens to a new article that is exactly above threshold (~0.79)
    new_art_above = Article(
        id=uuid.uuid4(),
        source_id=source.id,
        title="Title 4",
        url="http://test.com/4",
        content="raw",
        cleaned_content=above_text,
        content_hash="hash4"
    )
    
    canonical_id = detector.find_and_insert(new_art_above)
    assert canonical_id == art1.id  # Matches art1 because it is above 0.75 threshold
    
    # Deliberate near-miss: similarity is ~0.72 which is just below 0.75
    near_miss_art = Article(
        id=uuid.uuid4(),
        source_id=source.id,
        title="Title 6",
        url="http://test.com/6",
        content="raw",
        cleaned_content=below_text,
        content_hash="hash6"
    )
    assert detector.find_and_insert(near_miss_art) is None


async def test_duplicate_candidates_are_verified_and_retries_are_idempotent(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector = NearDuplicateDetector(db_session)
    await detector.initialize()
    first = Article(id=uuid.uuid4(), cleaned_content="Mars mission")
    unrelated = Article(id=uuid.uuid4(), cleaned_content="Ocean warming")
    assert detector.find_and_insert(first) is None
    # Force an approximate-index false positive: it must not merge sources.
    monkeypatch.setattr(detector.lsh, "query", lambda _: [str(first.id)])
    assert detector.find_and_insert(unrelated) is None
    assert detector.find_and_insert(first) is None
    duplicate = Article(id=uuid.uuid4(), cleaned_content="MARS mission")
    assert detector.find_and_insert(duplicate) == first.id
    assert len(detector.article_map) == 3


async def test_orchestrator_resilience(db_session: AsyncSession, monkeypatch):
    source = Source(name="Test", type=SourceTypeEnum.rss, url_or_identifier="test")
    db_session.add(source)
    await db_session.commit()
    
    a1 = Article(
        source_id=source.id, title="Good", url="g1", 
        content="This is a good article.", content_hash="h1"
    )
    a2 = Article(
        source_id=source.id, title="Bad", url="b1", 
        content="Bad", content_hash="h2"
    )
    db_session.add_all([a1, a2])
    await db_session.commit()
    
    # Mock clean_text to crash on the "Bad" article
    original_clean = clean_text
    def mock_clean(text):
        if text == "Bad":
            raise ValueError("Simulated crash")
        return original_clean(text)
        
    import app.services.preprocessing_orchestrator
    monkeypatch.setattr(app.services.preprocessing_orchestrator, "clean_text", mock_clean)
    
    orchestrator = PreprocessingOrchestrator()
    summary = await orchestrator.run_preprocessing_cycle(db_session)
    
    assert summary.total_processed == 2
    assert summary.status_counts[ProcessingStatusEnum.processed.value] == 1
    assert summary.status_counts[ProcessingStatusEnum.failed.value] == 1
    
    # Check DB
    result = await db_session.execute(select(Article).where(Article.url == "b1"))
    bad_art = result.scalar_one()
    assert bad_art.processing_status == ProcessingStatusEnum.failed
    
    result = await db_session.execute(select(Article).where(Article.url == "g1"))
    good_art = result.scalar_one()
    assert good_art.processing_status == ProcessingStatusEnum.processed


async def test_orchestrator_idempotency(db_session: AsyncSession):
    orchestrator = PreprocessingOrchestrator()
    summary1 = await orchestrator.run_preprocessing_cycle(db_session)
    assert summary1.total_processed == 0
    
    summary2 = await orchestrator.run_preprocessing_cycle(db_session)
    assert summary2.total_processed == 0


async def test_trigger_endpoint_admin(async_client):
    from app.api.deps import get_current_user
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: User(id=uuid.uuid4(), email="admin@test.com", role=RoleEnum.admin)
    
    response = await async_client.post("/api/v1/preprocessing/trigger")
    
    app.dependency_overrides.pop(get_current_user, None)
    
    assert response.status_code == 200
    assert "summary" in response.json()


async def test_trigger_endpoint_non_admin(async_client):
    from app.api.deps import get_current_user
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: User(id=uuid.uuid4(), email="user@test.com", role=RoleEnum.user)
    
    response = await async_client.post("/api/v1/preprocessing/trigger")
    
    app.dependency_overrides.pop(get_current_user, None)
    
    assert response.status_code == 403


async def test_status_endpoint_auth(async_client):
    from app.api.deps import get_current_user
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: User(id=uuid.uuid4(), email="user@test.com", role=RoleEnum.user)
    
    response = await async_client.get("/api/v1/preprocessing/status")
    
    app.dependency_overrides.pop(get_current_user, None)
    
    assert response.status_code == 200
    assert "status_counts" in response.json()


async def test_status_endpoint_unauth(async_client):
    response = await async_client.get("/api/v1/preprocessing/status")
    assert response.status_code == 401


@pytest.fixture(autouse=True)
def isolated_graph_projection(monkeypatch: pytest.MonkeyPatch):
    """Pipeline unit tests do not contact a live Neo4j projection."""
    from unittest.mock import AsyncMock

    from app.services.evolution.neo4j_writer import Neo4jWriter
    monkeypatch.setattr(Neo4jWriter, "sync_persisted_graph", AsyncMock(return_value={"success": True}))
