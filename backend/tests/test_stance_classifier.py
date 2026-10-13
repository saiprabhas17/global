"""Mocked three-class models only: no model weights or downloads are needed."""

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from app.services.verdict import engine


@pytest.fixture(autouse=True)
def reset_stance_state(monkeypatch):
    engine.STANCE_CACHE.clear()
    monkeypatch.setattr(engine, "_nli_tok", None)
    monkeypatch.setattr(engine, "_nli_mdl", None)
    monkeypatch.setattr(engine, "_nli_model_key", None)
    yield
    engine.STANCE_CACHE.clear()


def mock_classifier(monkeypatch, logits, labels=None, token_count=4) -> Any:
    labels = labels or {0: "contradiction", 1: "entailment", 2: "neutral"}
    tokenizer = Mock(
        return_value={"input_ids": torch.ones((1, token_count), dtype=torch.long)}
    )
    tokenizer.model_max_length = 512
    model = Mock(
        return_value=SimpleNamespace(logits=torch.tensor([logits], dtype=torch.float32))
    )
    model.config = SimpleNamespace(
        id2label=labels, num_labels=3, max_position_embeddings=512
    )
    monkeypatch.setattr(engine, "_load_nli", lambda config=None: (tokenizer, model))
    return tokenizer, model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "logits,expected",
    [
        ([0.0, 8.0, 0.0], "yes"),
        ([8.0, 0.0, 0.0], "no"),
        ([0.0, 0.0, 8.0], "neutral"),
    ],
)
async def test_actual_three_way_semantics_preserve_legacy_contract(
    monkeypatch, logits, expected
):
    tokenizer, model = mock_classifier(monkeypatch, logits)
    claim = "The council approved the bridge."
    premise = "The council rejected the bridge."
    assert await engine.nli_stance(claim, premise) == expected
    tokenizer.assert_called_once_with(
        premise, claim, return_tensors="pt", truncation=False
    )
    model.assert_called_once()


@pytest.mark.asyncio
async def test_neutral_does_not_mean_no_or_contradiction(monkeypatch):
    mock_classifier(monkeypatch, [0.0, 0.0, 9.0])
    result = await engine.classify_stance(
        "The bridge opened.", "The council meets tomorrow."
    )
    assert result.stance == "neutral"
    assert (
        await engine.nli_stance("The bridge opened.", "The council meets tomorrow.")
        == "neutral"
    )


@pytest.mark.asyncio
async def test_explicit_model_mapping_not_assumed_logit_order(monkeypatch):
    mock_classifier(
        monkeypatch,
        [8.0, 0.0, 0.0],
        {"0": "ENTAILMENT", "1": "neutral", "2": "contradiction"},
    )
    assert (
        await engine.nli_stance("The bridge opened.", "The bridge opened yesterday.")
        == "yes"
    )


@pytest.mark.asyncio
async def test_unknown_label_mapping_abstains(monkeypatch):
    _, model = mock_classifier(
        monkeypatch, [9.0, 0.0, 0.0], {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    )
    result = await engine.classify_stance("The bridge opened.", "The bridge is closed.")
    assert result.stance == "neutral"
    assert result.abstention_reason == "unknown_label_mapping"
    model.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_label2id_mapping_can_supply_semantics(monkeypatch):
    _, model = mock_classifier(
        monkeypatch, [0.0, 9.0, 0.0], {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    )
    model.config.label2id = {"contradiction": 0, "entailment": 1, "neutral": 2}
    assert (
        await engine.nli_stance("The bridge opened.", "The bridge opened yesterday.")
        == "yes"
    )


@pytest.mark.asyncio
async def test_low_confidence_abstains_instead_of_guessing(monkeypatch):
    mock_classifier(monkeypatch, [0.0, 0.4, 0.0])
    result = await engine.classify_stance(
        "The bridge opened.", "The bridge may have opened."
    )
    assert result.stance == "neutral"
    assert result.abstention_reason == "low_confidence"


@pytest.mark.asyncio
async def test_long_input_is_not_silently_truncated(monkeypatch):
    tokenizer, model = mock_classifier(monkeypatch, [0.0, 9.0, 0.0], token_count=513)
    claim = "The bridge opened. " * 40 + "Actually it did not open."
    result = await engine.classify_stance(claim, "The bridge remains closed.")
    assert result.stance == "neutral"
    assert result.abstention_reason == "input_exceeds_model_context"
    assert tokenizer.call_args.args[1] == claim
    model.assert_not_called()


@pytest.mark.asyncio
async def test_nonfinite_scores_abstain(monkeypatch):
    mock_classifier(monkeypatch, [float("nan"), 1.0, 0.0])
    result = await engine.classify_stance("The bridge opened.", "The bridge is closed.")
    assert result.stance == "neutral"
    assert result.abstention_reason == "invalid_model_scores"


@pytest.mark.asyncio
async def test_cache_hashes_full_ordered_text_not_prefixes(monkeypatch):
    infer = Mock(
        side_effect=lambda claim, other, config: engine.StanceResult(
            "contradiction" if claim.endswith("closed") else "entailment"
        )
    )
    monkeypatch.setattr(engine, "_run_nli", infer)
    prefix = "An identical prefix " * 20
    assert await engine.nli_stance(prefix + "open", "Evidence") == "yes"
    assert await engine.nli_stance(prefix + "closed", "Evidence") == "no"
    await engine.nli_stance(prefix + "open", "Evidence changed after the prefix")
    await engine.nli_stance("Evidence", prefix + "open")
    assert infer.call_count == 4
    await engine.nli_stance(prefix + "open", "Evidence")
    assert infer.call_count == 4
    assert all(len(key) == 64 for key in engine.STANCE_CACHE)


@pytest.mark.asyncio
async def test_cache_is_bounded_lru_and_expires(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(engine, "monotonic", lambda: now[0])
    monkeypatch.setattr(engine, "STANCE_CACHE_MAX", 2)
    monkeypatch.setattr(engine, "STANCE_CACHE_TTL_SECONDS", 5.0)
    infer = Mock(return_value=engine.StanceResult("neutral"))
    monkeypatch.setattr(engine, "_run_nli", infer)
    await engine.classify_stance("one", "premise")
    await engine.classify_stance("two", "premise")
    await engine.classify_stance("one", "premise")
    await engine.classify_stance("three", "premise")
    assert len(engine.STANCE_CACHE) == 2
    assert infer.call_count == 3
    await engine.classify_stance("two", "premise")  # Evicted, not one.
    assert infer.call_count == 4
    now[0] += 6.0
    await engine.classify_stance("two", "premise")
    assert infer.call_count == 5
    assert len(engine.STANCE_CACHE) <= 2


@pytest.mark.asyncio
async def test_model_revision_and_threshold_are_part_of_cache_key(monkeypatch):
    cfg = SimpleNamespace(
        NLI_MODEL="model-one", NLI_MODEL_REVISION="rev-one", NLI_CONFIDENCE_MIN=0.7
    )
    monkeypatch.setattr(engine, "get_settings", lambda: cfg)
    infer = Mock(return_value=engine.StanceResult("neutral"))
    monkeypatch.setattr(engine, "_run_nli", infer)
    await engine.classify_stance("claim", "premise")
    cfg.NLI_MODEL_REVISION = "rev-two"
    await engine.classify_stance("claim", "premise")
    cfg.NLI_MODEL = "model-two"
    await engine.classify_stance("claim", "premise")
    cfg.NLI_CONFIDENCE_MIN = 0.9
    await engine.classify_stance("claim", "premise")
    assert infer.call_count == 4


def test_lazy_loader_uses_three_class_local_weights_and_optional_revision(monkeypatch):
    tokenizer_loader = Mock(return_value=Mock())
    model = Mock()
    model_loader = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForSequenceClassification=SimpleNamespace(
                from_pretrained=model_loader
            ),
        ),
    )
    assert tokenizer_loader.call_count == 0
    config: engine.NLIConfiguration = {
        "model": "local-nli",
        "revision": "pinned-revision",
        "confidence_min": 0.7,
    }
    engine._load_nli(config)
    engine._load_nli(config)
    tokenizer_loader.assert_called_once_with(
        "local-nli", local_files_only=True, revision="pinned-revision"
    )
    model_loader.assert_called_once_with(
        "local-nli", local_files_only=True, revision="pinned-revision"
    )
    model.eval.assert_called_once()


@pytest.mark.asyncio
async def test_unavailable_model_is_actionable_and_not_cached(monkeypatch):
    monkeypatch.setattr(
        engine,
        "_load_nli",
        Mock(
            side_effect=engine.NLIUnavailableError(
                "Pre-cache NLI_MODEL and dependencies."
            )
        ),
    )
    with pytest.raises(engine.NLIUnavailableError, match="Pre-cache"):
        await engine.nli_stance("The bridge opened.", "The bridge is closed.")
    assert not engine.STANCE_CACHE


@pytest.mark.asyncio
async def test_invalid_confidence_config_is_actionable(monkeypatch):
    monkeypatch.setattr(
        engine, "get_settings", lambda: SimpleNamespace(NLI_CONFIDENCE_MIN=float("nan"))
    )
    with pytest.raises(engine.NLIUnavailableError, match="NLI_CONFIDENCE_MIN"):
        await engine.classify_stance("claim", "premise")
