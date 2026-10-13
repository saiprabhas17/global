"""HuggingFaceLLMClient: sentence filtering, claim scoring, batched extract."""
from typing import Any

from app.services.nlp.llm_client import (
    HuggingFaceLLMClient,
    _NoOpLLMClient,
    get_llm_client,
)


class FakeTensor(list[list[int]]):
    """Mimics torch.Tensor: a list subclass whose .to(device) is a no-op.

    Wraps the whole 2D batch (list-of-rows), matching how real HF
    tokenizers return a single BatchEncoding of tensors, not lists.
    """

    def to(self, device: str) -> "FakeTensor":
        return self


class FakeTokenizer:
    def __call__(self, prompts, return_tensors="pt", **kw):
        rows = [[1] * 8 for _ in prompts]
        return {
            "input_ids": FakeTensor(rows),
            "attention_mask": FakeTensor(rows),
        }

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["yes"] * len(ids)


class FakeModel:
    def generate(self, **kw: Any) -> Any:
        return kw["input_ids"]


def test_split_sentences_keeps_short_facts_and_long_sentences():
    c = HuggingFaceLLMClient()
    text = (
        "Earth is round. "
        "This is a perfectly reasonable declarative sentence with several words and no drama. "  # kept
        "The agency reported " + ("word " * 120)
    )
    out = c._split_sentences(text)
    assert len(out) == 3
    assert out[0] == "Earth is round."
    assert out[-1].startswith("The agency reported")


async def test_extract_claims_processes_late_article_sentences():
    c = HuggingFaceLLMClient()
    c._pipe = (FakeTokenizer(), FakeModel(), "cpu")
    sentences = [f"The agency reported {index} incidents in its annual records."
                 for index in range(19)]
    claims = await c.extract_claims(" ".join(sentences))
    assert [claim.claim_text for claim in claims] == sentences


async def test_model_loading_runs_off_event_loop(monkeypatch):
    import threading

    loop_thread = threading.get_ident()
    load_threads = []
    c = HuggingFaceLLMClient()

    def load():
        load_threads.append(threading.get_ident())
        c._pipe = (FakeTokenizer(), FakeModel(), "cpu")

    monkeypatch.setattr(c, "_load", load)
    await c.extract_claims("The agency reported 20 incidents in its annual records.")
    assert load_threads and load_threads[0] != loop_thread


async def test_local_mutation_summary_describes_actual_changes():
    c = HuggingFaceLLMClient()
    summary = await c.generate_mutation_summary([
        "The agency reported 10 incidents.", "The agency reported 20 incidents.",
    ])
    assert "10 → 20" in summary
    assert "does not establish truth or transmission" in summary
    assert "API key" not in summary


def test_score_claim_rewards_verifiable_language():
    c = HuggingFaceLLMClient()
    factual = "The agency reported a 15 percent increase in 2023 according to records."
    subjective = "I think this whole thing feels ridiculous and I hate it a lot, honestly speaking."
    assert c._score_claim(factual) >= 0.6
    assert c._score_claim(subjective) < 0.4


async def test_extract_claims_accepts_factual_rejects_subjective():
    c = HuggingFaceLLMClient()
    class ClassifiedTokenizer(FakeTokenizer):
        def batch_decode(self, ids, skip_special_tokens=True):
            return ["factual", "opinion"]

    c._pipe = (ClassifiedTokenizer(), FakeModel(), "cpu")
    text = (
        "The agency reported a 15 percent increase in March according to official records. "
        "I think this whole situation feels ridiculous and personally I hate it, honestly. "
    )
    claims = await c.extract_claims(text)
    assert len(claims) == 1
    assert claims[0].claim_text.startswith("The agency reported")
    assert claims[0].confidence is not None and claims[0].confidence >= 0.5


async def test_noop_client_returns_empty_and_factory_respects_backend(monkeypatch):
    noop = _NoOpLLMClient()
    assert await noop.extract_claims("anything") == []

    # The factory reads config through module-level get_settings(); stub it
    # with a lightweight namespace (Settings has no LLM_BACKEND field).
    from types import SimpleNamespace

    from app.services.nlp import llm_client as lc

    monkeypatch.setattr(
        lc,
        "get_settings",
        lambda: SimpleNamespace(ANTHROPIC_API_KEY="", LLM_BACKEND="none"),
    )
    assert isinstance(get_llm_client(), _NoOpLLMClient)


async def test_local_supplied_evidence_check_has_no_paid_provider_requirement(monkeypatch):
    from unittest.mock import AsyncMock

    from app.services.verdict.engine import StanceResult

    stance = AsyncMock(return_value=StanceResult("entailment"))
    monkeypatch.setattr("app.services.verdict.engine.classify_stance", stance)
    result = await HuggingFaceLLMClient().check_external_claim("The bridge opened.", "The bridge opened.")
    assert result["verdict"] == "SUPPORTED"
    stance.assert_awaited_once_with("The bridge opened.", "The bridge opened.")
