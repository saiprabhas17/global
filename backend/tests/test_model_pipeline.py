"""Explicit real-model smoke: engineering behavior, never an accuracy benchmark."""
import pytest

from app.services.nlp.llm_client import HuggingFaceLLMClient
from app.services.verdict.engine import classify_stance

pytestmark = pytest.mark.model


async def test_real_local_stance_support_and_contradiction() -> None:
    supported = await classify_stance("The bridge is open.", "The bridge is open.")
    contradicted = await classify_stance(
        "The bridge is open.", "The bridge is closed and is not open."
    )
    assert supported.stance == "entailment" and supported.abstention_reason is None
    assert contradicted.stance == "contradiction" and contradicted.abstention_reason is None


async def test_real_local_extraction_retains_original_factual_sentence() -> None:
    sentence = "NASA launched the Artemis I mission on November 16, 2022."
    claims = await HuggingFaceLLMClient().extract_claims(sentence)
    assert any(claim.claim_text == sentence for claim in claims)
    assert all(claim.claim_text in sentence for claim in claims)
