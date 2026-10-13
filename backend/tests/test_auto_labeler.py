# mypy: disallow-untyped-defs=False, disallow-incomplete-defs=False
"""Exploratory machine-signal tests, not evidence of independent human gold.

Retains label/edge-case coverage, and checks engine-score independence,
automatic provenance, historical preservation and honest failure handling.
No heavy model downloads are needed for these unit tests.
"""

import re
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

# ──────────────────────────────────────────────────────────────────────
# Lexical annotator (no heavy model dependencies)
# ──────────────────────────────────────────────────────────────────────
from scripts.auto_label_eval_pairs import (
    ANNOTATOR_ENTITY,
    ANNOTATOR_LEXICAL,
    ANNOTATOR_NLI,
    _tokenize,
    lexical_label,
)


@pytest.fixture(autouse=True)
def text_only_entity_fixture(monkeypatch):
    def nlp(text):
        return SimpleNamespace(ents=[SimpleNamespace(text=value) for value in re.findall(r"\b[A-Z][a-z]+\b", text)])
    monkeypatch.setattr("scripts.auto_label_eval_pairs._load_spacy", lambda: nlp)


class TestLexicalAnnotator:
    def test_identical_text_is_same_story(self):
        text = "The World Health Organization declared a global health emergency on Monday."
        assert lexical_label(text, text, 0.95) == "SAME_STORY"

    def test_near_identical_paraphrase(self):
        a = "The WHO declared a global health emergency on Monday."
        b = "The World Health Organization announced a global health emergency Monday."
        # Text-only lexical overlap; the deprecated score must not affect the label.
        label = lexical_label(a, b, 0.85)
        assert label in ("SAME_STORY", "EVOLVED")

    def test_unrelated_claims_are_distinct(self):
        a = "Apple released the iPhone 16 with a new chip design."
        b = "Amazon rainforest fires reached record levels in September 2026."
        assert lexical_label(a, b, 0.15) == "DISTINCT"

    def test_empty_text_is_distinct(self):
        assert lexical_label("", "Some claim text here", 0.3) == "DISTINCT"

    def test_tokenizer_lowercases(self):
        tokens = _tokenize("The WORLD Health Organization")
        assert "the" in tokens
        assert "world" in tokens
        assert "THE" not in tokens


class TestEntityAnnotator:
    """Entity signal uses deterministic extracted-entity fixtures."""

    def test_no_entities_remains_distinct_at_low_score(self):
        from scripts.auto_label_eval_pairs import entity_label
        # Very generic text with no named entities + low sim
        a = "something happened somewhere at some point in time"
        b = "another thing occurred in a different place recently"
        label = entity_label(a, b, 0.20)
        assert label == "DISTINCT"

    def test_no_entities_high_sim(self):
        from scripts.auto_label_eval_pairs import entity_label
        a = "something happened somewhere"
        b = "something happened somewhere else"
        label = entity_label(a, b, 0.90)
        assert label == "DISTINCT"

    def test_shared_entities_high_sim(self):
        from scripts.auto_label_eval_pairs import entity_label
        a = "President Biden met with Prime Minister Starmer in London on October 1."
        b = "Biden and Starmer held a summit meeting in London on October 1."
        label = entity_label(a, b, 0.80)
        assert label in ("SAME_STORY", "EVOLVED")


class TestLabelValidation:
    """Ensure all annotators only produce valid labels."""

    VALID_LABELS = frozenset({"SAME_STORY", "EVOLVED", "DISTINCT"})

    def test_lexical_produces_valid_labels(self):
        pairs = [
            ("The cat sat on the mat.", "Dogs run in the park.", 0.3),
            ("Stock market crashed 500 points.", "Stock market fell 500 points today.", 0.8),
            ("same text", "same text", 0.99),
        ]
        for a, b, s in pairs:
            label = lexical_label(a, b, s)
            assert label in self.VALID_LABELS, f"Invalid label: {label}"


@pytest.mark.parametrize("texts", [
    ("Atlas announced a mission in London", "Atlas launched a mission in London"),
    ("unrelated generic first text", "another unrelated phrase"),
    ("", "Atlas said something"),
])
def test_entity_and_lexical_ignore_engine_score(texts):
    from scripts.auto_label_eval_pairs import entity_label

    for signal in (entity_label, lexical_label):
        labels = {signal(texts[0], texts[1], score) for score in (-1.0, 0.0, 0.59, 0.9, 1.0, float("nan"))}
        assert len(labels) == 1


def test_machine_failure_is_not_fabricated_distinct(monkeypatch):
    from scripts.auto_label_eval_pairs import label_pair

    def fail(*args):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr("scripts.auto_label_eval_pairs.nli_label", fail)
    pair = {"pair_id": "11111111-1111-1111-1111-111111111111", "text_a": "Atlas reported a mission", "text_b": "Atlas reported a mission", "sim_score": 0.9, "missing_annotators": {ANNOTATOR_NLI, ANNOTATOR_LEXICAL, ANNOTATOR_ENTITY}}
    votes = label_pair(pair)
    assert {v["annotator"] for v in votes} == {ANNOTATOR_LEXICAL, ANNOTATOR_ENTITY}
    assert all(v["origin"] == "automatic" for v in votes)


@pytest.mark.asyncio
async def test_save_labels_explicit_automatic_and_conflict_safe(monkeypatch):
    from scripts.auto_label_eval_pairs import save_labels

    db = AsyncMock()
    db.execute.return_value = SimpleNamespace(scalar_one_or_none=lambda: "inserted")
    context = AsyncMock()
    context.__aenter__.return_value = db
    monkeypatch.setattr("scripts.auto_label_eval_pairs.async_session_maker", lambda: context)
    votes = [{"pair_id": "11111111-1111-1111-1111-111111111111", "annotator": ANNOTATOR_LEXICAL, "label": "SAME_STORY", "origin": "human"}]
    assert await save_labels(votes) == 1
    statement = db.execute.call_args.args[0]
    compiled = statement.compile(dialect=cast(Any, postgresql.dialect)())
    assert compiled.params["origin"] == "automatic"  # Caller cannot upgrade a machine signal.
    assert compiled.params["provenance"]["independent_human"] is False
    assert "DO NOTHING" in str(compiled) and "DELETE" not in str(compiled)
    assert "uq_eval_label_pair_annotator_origin" in str(compiled)
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_automatic_writer_refuses_human_namespace(monkeypatch):
    from scripts.auto_label_eval_pairs import save_labels

    db = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = db
    monkeypatch.setattr("scripts.auto_label_eval_pairs.async_session_maker", lambda: context)
    with pytest.raises(ValueError, match="machine signals"):
        await save_labels([{"pair_id": "11111111-1111-1111-1111-111111111111", "annotator": "user:spoofed", "label": "DISTINCT"}])
    db.execute.assert_not_awaited()
