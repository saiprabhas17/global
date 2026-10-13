from unittest.mock import AsyncMock, patch

import pytest

from app.services.nlp.embedding_service import EmbeddingService
from app.services.nlp.entity_extractor import EntityExtractor
from app.services.nlp.llm_client import AnthropicLLMClient


def test_entity_extractor():
    EntityExtractor.load_model()
    extractor = EntityExtractor()
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    entities = extractor.extract_entities(text)
    
    # We should find Apple Inc (ORG), Steve Jobs (PERSON), Cupertino (GPE), California (GPE)
    entity_texts = [e.entity_text for e in entities]
    assert "Apple Inc." in entity_texts
    assert "Steve Jobs" in entity_texts
    assert "Cupertino" in entity_texts
    
    # Check offsets
    for entity in entities:
        assert text[entity.start_char:entity.end_char] == entity.entity_text

@pytest.mark.model
def test_embedding_service():
    EmbeddingService.load_model()
    service = EmbeddingService()
    text1 = "Climate change is a significant global challenge."
    text2 = "Climate change is a significant global challenge."
    text3 = "The quick brown fox jumps over the lazy dog."
    
    emb1 = service.generate_embedding(text1)
    emb2 = service.generate_embedding(text2)
    emb3 = service.generate_embedding(text3)
    
    # Check dimension
    assert len(emb1) == 768
    
    # Check determinism (identical texts should have identical embeddings)
    assert emb1 == emb2
    
    # Check uniqueness for different texts
    assert emb1 != emb3

@pytest.mark.asyncio
async def test_llm_client_malformed_json():
    # We must patch get_settings so AnthropicLLMClient can be instantiated with a dummy key
    with patch("app.services.nlp.llm_client.get_settings") as mock_get_settings, \
         patch("app.services.nlp.llm_client.AsyncAnthropic") as MockAsyncAnthropic:
         
        mock_settings = mock_get_settings.return_value
        mock_settings.ANTHROPIC_API_KEY = "dummy"
        mock_settings.LLM_MODEL = "test-model"
        
        # Setup the mock client
        mock_client_instance = AsyncMock()
        MockAsyncAnthropic.return_value = mock_client_instance
        
        # Create client
        client = AnthropicLLMClient()
        
        # Mock a malformed JSON response from Claude
        mock_message = AsyncMock()
        mock_content = AsyncMock()
        mock_content.text = "Not a valid JSON structure [{"
        mock_message.content = [mock_content]
        mock_client_instance.messages.create.return_value = mock_message
        
        # Should gracefully fail or raise ValueError on parsing failure
        # Actually in the code, it raises ValueError("Invalid LLM response format: ...")
        with pytest.raises(ValueError, match="Invalid LLM response format"):
            await client.extract_claims("Some text")
        
        # Test valid JSON but non-list structure (e.g. dict)
        mock_content.text = '{"claim": "single claim"}'
        with pytest.raises(ValueError, match="Expected a JSON array"):
            await client.extract_claims("Some text")
