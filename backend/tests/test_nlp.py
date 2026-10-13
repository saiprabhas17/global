from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.article import Article, NLPStatusEnum, ProcessingStatusEnum
from app.models.claim import Claim
from app.models.source import Source, SourceTypeEnum
from app.services.nlp.llm_client import ExtractedClaim
from app.services.nlp_orchestrator import NLPOrchestrator

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    # Mocking standard response
    client.extract_claims.return_value = [
        ExtractedClaim(claim_text="This is a test claim.", confidence=0.9)
    ]
    # Simulate client being configured
    client.client = True 
    return client

async def test_nlp_orchestrator_success(db_session, mock_llm_client):
    # Setup test source
    source = Source(name="Test Source", url_or_identifier="http://test.com", type=SourceTypeEnum.rss)
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)
    
    # Setup test article
    article = Article(
        title="Test Article",
        content="Test content. This is a test claim.",
        url="http://test.com", source_id=source.id, content_hash="hash3",
        processing_status=ProcessingStatusEnum.processed,
        nlp_status=NLPStatusEnum.pending,
        cleaned_content="Test content. This is a test claim."
    )
    db_session.add(article)
    await db_session.commit()
    await db_session.refresh(article)

    orchestrator = NLPOrchestrator(llm_client=mock_llm_client)
    
    # Mock entity and embedding extraction to avoid loading heavy models during tests
    with patch("app.services.nlp.entity_extractor.EntityExtractor.extract_entities") as mock_ner, \
         patch("app.services.nlp.embedding_service.EmbeddingService.generate_embedding") as mock_embed:
        
        mock_ner.return_value = []
        mock_embed.return_value = [0.1] * 768
        
        summary = await orchestrator.run_nlp_cycle(db_session)
        
    assert summary.total_processed == 1
    assert summary.claims_extracted == 1
    assert summary.llm_calls_made == 1
    assert summary.status_counts[NLPStatusEnum.completed.value] == 1
    
    # Check DB
    await db_session.refresh(article)
    assert article.nlp_status == NLPStatusEnum.completed
    assert article.embedding is not None
    assert len(article.embedding) == 768
    
    # Check claim exists
    from sqlalchemy import select
    res = await db_session.execute(select(Claim).where(Claim.article_id == article.id))
    claims = res.scalars().all()
    assert len(claims) == 1
    assert claims[0].claim_text == "This is a test claim."
    assert len(claims[0].embedding) == 768


async def test_nlp_orchestrator_empty_claims(db_session, mock_llm_client):
    source = Source(name="Test Source", url_or_identifier="http://test.com", type=SourceTypeEnum.rss)
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)
    
    # Setup test article
    article = Article(
        title="Opinion Piece",
        content="I think this is great.",
        url="http://test2.com", source_id=source.id, content_hash="hash4",
        processing_status=ProcessingStatusEnum.processed,
        nlp_status=NLPStatusEnum.pending,
        cleaned_content="I think this is great."
    )
    db_session.add(article)
    await db_session.commit()
    
    # Mock empty claims
    mock_llm_client.extract_claims.return_value = []
    
    orchestrator = NLPOrchestrator(llm_client=mock_llm_client)
    with patch("app.services.nlp.embedding_service.EmbeddingService.generate_document_embedding",
               return_value=[0.1] * 768):
        summary = await orchestrator.run_nlp_cycle(db_session, chain_evolution=False)
    
    assert summary.total_processed == 1
    assert summary.claims_extracted == 0
    assert summary.llm_calls_made == 1
    assert summary.status_counts[NLPStatusEnum.completed.value] == 1
    await db_session.refresh(article)
    assert article.embedding is not None


async def test_nlp_orchestrator_failure_recovery(db_session, mock_llm_client):
    source = Source(name="Test Source", url_or_identifier="http://test.com", type=SourceTypeEnum.rss)
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)
    
    article1 = Article(
        title="Article 1", content="Content 1", url="http://1.com", source_id=source.id, content_hash="hash1",
        processing_status=ProcessingStatusEnum.processed, nlp_status=NLPStatusEnum.pending
    )
    article2 = Article(
        title="Article 2", content="Content 2", url="http://2.com", source_id=source.id, content_hash="hash2",
        processing_status=ProcessingStatusEnum.processed, nlp_status=NLPStatusEnum.pending
    )
    db_session.add_all([article1, article2])
    await db_session.commit()
    
    # Make extract_claims fail on the first call, succeed on the second
    mock_llm_client.extract_claims.side_effect = [
        ValueError("Simulated LLM error"),
        [ExtractedClaim(claim_text="Success", confidence=0.8)]
    ]
    
    orchestrator = NLPOrchestrator(llm_client=mock_llm_client)
    
    with patch("app.services.nlp.entity_extractor.EntityExtractor.extract_entities") as mock_ner, \
         patch("app.services.nlp.embedding_service.EmbeddingService.generate_embedding") as mock_embed:
        mock_ner.return_value = []
        mock_embed.return_value = [0.1] * 768
        summary = await orchestrator.run_nlp_cycle(db_session)
        
    assert summary.total_processed == 2
    assert summary.status_counts[NLPStatusEnum.failed.value] == 1
    assert summary.status_counts[NLPStatusEnum.completed.value] == 1


async def test_nlp_orchestrator_cost_cap(db_session, mock_llm_client):
    source = Source(name="Test Source", url_or_identifier="http://test.com", type=SourceTypeEnum.rss)
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)
    
    # Create 15 articles, cap is 10
    articles = [
        Article(
            title=f"Article {i}", content=f"Content {i}", url=f"http://{i}.com", source_id=source.id, content_hash=f"hash{i}",
            processing_status=ProcessingStatusEnum.processed, nlp_status=NLPStatusEnum.pending
        )
        for i in range(15)
    ]
    db_session.add_all(articles)
    await db_session.commit()
    
    orchestrator = NLPOrchestrator(llm_client=mock_llm_client)
    orchestrator.max_articles = 10  # Explicitly set
    
    with patch("app.services.nlp.entity_extractor.EntityExtractor.extract_entities") as mock_ner, \
         patch("app.services.nlp.embedding_service.EmbeddingService.generate_embedding") as mock_embed:
        mock_ner.return_value = []
        mock_embed.return_value = [0.1] * 768
        summary = await orchestrator.run_nlp_cycle(db_session)
        
    assert summary.total_processed == 10
    assert summary.llm_calls_made == 10
    
    # 5 should remain pending
    from sqlalchemy import func, select
    res = await db_session.execute(select(func.count(Article.id)).where(Article.nlp_status == NLPStatusEnum.pending))
    pending_count = res.scalar()
    assert pending_count == 5


async def test_nlp_orchestrator_llm_failure_marks_failed(db_session):
    """When the LLM client raises (e.g. backend down), the article is
    marked failed — not silently skipped. The old 'missing key -> skip'
    path was removed when the HuggingFace fallback landed."""
    from unittest.mock import AsyncMock

    source = Source(name="Test Source", url_or_identifier="http://test.com", type=SourceTypeEnum.rss)
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)

    article = Article(
        title="Article", content="Real factual content about the world.", url="http://x.com", source_id=source.id, content_hash="hashx",
        processing_status=ProcessingStatusEnum.processed, nlp_status=NLPStatusEnum.pending
    )
    db_session.add(article)
    await db_session.commit()

    mock_llm_client = MagicMock()
    mock_llm_client.extract_claims = AsyncMock(side_effect=ValueError("LLM unavailable"))

    orchestrator = NLPOrchestrator(llm_client=mock_llm_client)
    summary = await orchestrator.run_nlp_cycle(db_session)

    assert summary.total_processed == 1
    assert summary.status_counts[NLPStatusEnum.failed.value] == 1
    assert summary.llm_calls_made == 0


@pytest.fixture(autouse=True)
def isolated_graph_projection(monkeypatch: pytest.MonkeyPatch):
    """Pipeline unit tests do not contact a live Neo4j projection."""
    from unittest.mock import AsyncMock

    from app.services.evolution.neo4j_writer import Neo4jWriter
    monkeypatch.setattr(Neo4jWriter, "sync_persisted_graph", AsyncMock(return_value={"success": True}))
